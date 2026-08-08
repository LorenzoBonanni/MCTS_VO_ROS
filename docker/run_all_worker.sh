#!/usr/bin/env bash
set -e

if [[ -z "${SLURM_ARRAY_TASK_ID}" || -z "${SLURM_PROCID}" ]]; then
    echo "requires SLURM_ARRAY_TASK_ID and SLURM_PROCID" >&2
    exit 1
fi

task_id=$(( SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID ))

# Grid arrays. Parsed BEFORE total_tasks is computed: that count used to be
# hard-coded as 3 * 2 * NUM_SEEDS, so adding a scene or an algorithm silently
# truncated the campaign instead of extending it.
IFS=' ' read -ra ALGO_ARR   <<< "${ALGORITHMS:-MCTS VO-TREE VO-PLANNER}"
IFS=' ' read -ra TRAJ_ARR   <<< "${TRAJECTORIES:-sinusoidal_complex intention_complex}"
N_SEEDS=${NUM_SEEDS:-30}

N_ALGO=${#ALGO_ARR[@]}
N_TRAJ=${#TRAJ_ARR[@]}
total_tasks=$(( N_ALGO * N_TRAJ * N_SEEDS ))

if (( task_id >= total_tasks )); then
    echo "task_id $task_id beyond total $total_tasks – nothing to do"
    exit 0
fi

seed=$(( task_id % N_SEEDS ))
IDX=$(( task_id / N_SEEDS ))
traj_idx=$(( IDX % N_TRAJ ))
algo_idx=$(( IDX / N_TRAJ ))

ALGO="${ALGO_ARR[$algo_idx]}"
SCENE="${TRAJ_ARR[$traj_idx]}"

# Per-scene parameters, from the 192-cell / 9600-run sweep. The two scene
# families behave differently enough that a single setting is wrong for one of
# them, so each takes its own. gamma is the dominant axis in both: on
# sinusoidal_complex the goal rate falls 21.2% -> 0.1% going from 0.65 to 0.95,
# with timeouts rising to 56%, so a short horizon wins clearly.
#
# sinusoidal*: rs=1.4 g=0.65 c=5.0 mov=0.25 gave 36% goal and 0% voluntary
#   collisions. The grid's best single cell was 50% at rs=1.2 mov=0.2, but that
#   is the maximum of 96 noisy estimates at n=50 (so inflated by selection) and
#   it sits at the least safe rs with no margin above the true obstacle speed.
#   The whole gamma=0.65 c=5.0 region averages 33.3% goal (CI 28-39) with 1
#   voluntary collision in 300 runs, and this cell is representative of it.
#
# intention*: no configuration works - the best cells reach 2%, i.e. one run in
#   fifty, indistinguishable from zero across 48 configurations and 2400 runs.
#   These values are therefore chosen on the marginals rather than on a winning
#   cell: gamma=0.65 as above, rs=1.6 because voluntary collisions fall
#   monotonically with rs (2.9 / 2.4 / 1.8 % for 1.2 / 1.4 / 1.6), c=2.0 as one
#   of the two joint-best cells, mov=0.25 for headroom over the 0.2 m/s
#   obstacles. Expect ~0% goal: the scene is a Section 6 immobilisation case.
case "$SCENE" in
    sinusoidal*) RS=1.4; GAMMA=0.65; C=5.0;  MOV=0.25 ;;
    intention*)  RS=1.6; GAMMA=0.65; C=2.0;  MOV=0.25 ;;
    *) echo "no tuned parameters for scene '$SCENE'" >&2; exit 1 ;;
esac
# Overridable, but the defaults above are the tuned ones.
RS="${RADIUS_SCALE:-$RS}"
GAMMA="${GAMMA_PER_SECOND:-$GAMMA}"
C="${EXPLORATION_C:-$C}"
MOV="${MAX_OBS_VEL:-$MOV}"

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
# Shared across every task on this node, deliberately. Here one task is a
# single run, so a per-task cache meant every run recompiled the jitted kernels
# from scratch - about 15 s each, and 360 times over the campaign. /tmp is
# node-local, so this is warm for every array task that lands on a node after
# the first. Numba writes cache entries via atomic rename, so concurrent use is
# safe; the stagger below keeps the eight procs of the first task from all
# compiling the same kernels at once.
export NUMBA_CACHE_DIR="/tmp/numba-shared"
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

# Stagger the eight procs sharing this node so they do not all JIT-compile into
# the empty shared cache simultaneously. Whoever gets there first populates it;
# the rest find it warm. Costs at most 14 s once per node, and only when the
# cache is cold.
sleep $(( SLURM_PROCID * 2 ))

echo "task $task_id: ${ALGO} ${SCENE} seed ${seed} (rs=${RS} gamma=${GAMMA} c=${C} mov=${MOV})"

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
        --max-obs-vel "$MOV" \
        --max-obs-radius "${MAX_OBS_RADIUS:-0.5}" \
        --vo-geometry "${VO_GEOMETRY:-paper}" \
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
