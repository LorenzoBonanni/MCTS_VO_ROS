#!/usr/bin/env bash
set -e

if [[ -z "${SLURM_ARRAY_TASK_ID}" || -z "${SLURM_PROCID}" ]]; then
    echo "requires SLURM_ARRAY_TASK_ID and SLURM_PROCID" >&2
    exit 1
fi

task_id=$(( SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID ))
total_tasks=$(( 3 * 2 * NUM_SEEDS ))                    # algos * scenes * seeds

if (( task_id >= total_tasks )); then
    echo "task_id $task_id beyond total $total_tasks – nothing to do"
    exit 0
fi

# Grid arrays
IFS=' ' read -ra ALGO_ARR   <<< "${ALGORITHMS:-MCTS VO-TREE VO-PLANNER}"
IFS=' ' read -ra TRAJ_ARR   <<< "${TRAJECTORIES:-sinusoidal_complex intention_complex}"
N_SEEDS=${NUM_SEEDS:-30}

N_ALGO=${#ALGO_ARR[@]}
N_TRAJ=${#TRAJ_ARR[@]}

seed=$(( task_id % N_SEEDS ))
IDX=$(( task_id / N_SEEDS ))
traj_idx=$(( IDX % N_TRAJ ))
algo_idx=$(( IDX / N_TRAJ ))

ALGO="${ALGO_ARR[$algo_idx]}"
SCENE="${TRAJ_ARR[$traj_idx]}"

# Fixed parameters
RS="${RADIUS_SCALE:-1.8}"
GAMMA="${GAMMA_PER_SECOND:-0.81}"
C="${EXPLORATION_C:-1.0}"

# Output directory – directly in the repository's debug/ folder
DEBUG_ROOT="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}/mctsVoRos/debug"
SCENE_DIR="${DEBUG_ROOT}/${SCENE}"
mkdir -p "${SCENE_DIR}"

# Logs stored separately
LOG_DIR="/root/campaign/logs/${SCENE}"
mkdir -p "${LOG_DIR}"

CSV="${SCENE_DIR}/data_${ALGO}_${seed}.csv"
LOG="${LOG_DIR}/${ALGO}_${seed}.log"

if [[ -f "${CSV}" ]]; then
    echo "task $task_id: ${ALGO} ${SCENE} seed ${seed} already done"
    exit 0
fi

# ---------- ROS & environment ----------
set +eu
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -e

# Unique isolation per task
export ROS_DOMAIN_ID=$(( task_id % 101 ))
export ROS_LOCALHOST_ONLY=1
export NUMBA_CACHE_DIR="/tmp/numba-${task_id}"
export MPLCONFIGDIR="/tmp/mpl-${task_id}"
export HOME="/tmp/home-${task_id}"
export ROS_HOME="$HOME/.ros"
export ROS_LOG_DIR="$ROS_HOME/log"
export XDG_CONFIG_HOME="$HOME/.config"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$ROS_LOG_DIR" "$XDG_CONFIG_HOME"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"

# ---------- Create isolated working directory with correct structure ----------
# We mimic the original sweep: a cell directory containing the env_build symlink,
# and a work subdirectory where the Python script actually runs.
CELL_DIR="/tmp/campaign-cell-${task_id}"
WORK="${CELL_DIR}/work"
mkdir -p "$WORK"

REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

# env_build must be one level above the working directory, because
# loopHandler_copy.py uses a relative path: ../env_build/...
ln -sfn "$REPO/env_build" "$CELL_DIR/env_build"

# Symlink the project code into the work directory
ln -sfn "$REPO/mctsVoRos/MCTS_VO" "$WORK/MCTS_VO"
for f in "$REPO"/mctsVoRos/*.py; do
    ln -sfn "$f" "$WORK/$(basename "$f")"
done

# Point the local debug/ to the shared repository debug folder
ln -sfn "$DEBUG_ROOT" "$WORK/debug"

cd "$WORK"

echo "task $task_id: ${ALGO} ${SCENE} seed ${seed} (rs=${RS} gamma=${GAMMA} c=${C})"

# Run the experiment (retry once if needed)
for attempt in 1 2; do
    echo "=== attempt $attempt ===" >> "$LOG"
    set +e
    python3 loopHandler_copy.py \
        --algorithm "$ALGO" \
        --trajectories "$SCENE" \
        --exp_num "$seed" \
        --env-render headless \
        --no-plots \
        --radius-scale "$RS" \
        --gamma-per-second "$GAMMA" \
        --exploration-c "$C" \
        --max-obs-vel "${MAX_OBS_VEL:-0.25}" \
        >> "$LOG" 2>&1
    rc=$?
    set -e
    echo "=== exit status $rc ===" >> "$LOG"
    [[ -f "$CSV" ]] && break
    echo "attempt $attempt failed (exit $rc)" >&2
done

if [[ ! -f "$CSV" ]]; then
    echo "task $task_id: NO DATA after 2 attempts – see $LOG" >&2
    exit 1
fi

# Kill Unity (by unique working directory)
for p in $(pgrep -f 'env_build/.*x86_64' 2>/dev/null); do
    if [[ "$(readlink -f /proc/$p/cwd 2>/dev/null)" == "$(readlink -f "$WORK")" ]]; then
        kill "$p" 2>/dev/null || true
    fi
done

# Cleanup temporary cell and work directories
rm -rf "$CELL_DIR"

echo "task $task_id: done"
