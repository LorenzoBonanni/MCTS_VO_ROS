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

# Per-scene parameters. gamma dominates everything else, and the values below
# are far lower than the 0.65 used until now, which was itself the floor of the
# original grid rather than an optimum.
#
# Why: the search holds obstacles at their observed position (paper 4.2.1), so
# it is only usable while they have not moved far. The effective horizon is
# dt/(1 - gamma^dt) and the condition is v_closing * horizon < r_R + r_i, i.e.
# 0.15 + 0.10 = 0.25 m. At gamma=0.65 the horizon is 23.7 steps and a 0.2 m/s
# obstacle covers 47 cm - nearly twice the whole margin - so "stand still"
# scores as risk-free while being the most dangerous action available. That is
# what produced 68-96% obstacle-initiated collisions in the 600-run campaign.
#
# The gamma sweep (0.02-0.65, 2 rs x 2 c x 2 scenes) found a plateau on each
# scene, ending where the validity condition is violated:
#
#   gamma/s           0.02 0.04 0.07 0.10 0.15 0.20 0.30 0.43 0.55 0.65
#   moves/margin       25%  29%  34%  39%  46%  54%  71%  99% 138% 190%
#   intention_complex  88%  85%  86%  91%  88%  92%  88%   6%   0%   0%
#   sinusoidal_complex 80%  83%  79%  78%  56%  49%  12%   3%   3%  26%
#
# intention tolerates up to 71% of the margin, sinusoidal only ~39% - four
# obstacles rather than two means more simultaneous encounters. Both are flat
# below their boundary, so gamma is chosen mid-plateau rather than at its best
# cell: the per-cell maxima are selection artefacts, and the one we tested
# out-of-sample confirmed it (84% at gamma=0.10 became 74% on fresh seeds, 79%
# over 100). Expect roughly 80% on sinusoidal_complex and 88% on
# intention_complex, not the 87% / 96% the best cells report.
#
# 0.04 rather than 0.10 for sinusoidal* because 0.10 is the last point before
# the drop to 56%, and sitting on a boundary is how a scene change turns into a
# collapse. 0.30 for intention* is likewise inside its plateau, not at the edge.
#
# The easy scenes were not swept. Their obstacles run at 0.1 m/s rather than
# 0.2, so the same gamma leaves the model comfortably inside its validity
# region - the risk there is a horizon that is shorter than it needs to be,
# not one that is too long.
case "$SCENE" in
    sinusoidal*) RS=1.4; GAMMA=0.04; C=5.0;  MOV=0.25 ;;
    intention*)  RS=1.4; GAMMA=0.30; C=5.0;  MOV=0.25 ;;
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

# Resume, but only over a run made with THESE parameters. The check used to be
# a bare [[ -f "$CSV" ]], and the filename carries only algorithm and seed, so a
# leftover from any earlier experiment counted as "already done". That happened:
# a campaign re-run at c=5.0 skipped 26 of 30 seeds still on disk from a c=1.0
# run, and the summariser then averaged the mixture. It looked like the same
# configuration producing 7% and 37% goal on different days, and cost a day of
# chasing a nondeterminism that was not there. A mismatched CSV is overwritten,
# not skipped: the parameters this task was given are the intended ones.
params_match() {
    awk -F, -v want="$RS|$GAMMA|$C|$MOV|${MAX_OBS_RADIUS:-0.5}|${VO_GEOMETRY:-paper}" '
        NR == 1 { for (i = 1; i <= NF; i++) h[$i] = i; next }
        NR == 2 {
            split(want, w, "|")
            n = split("radiusScale gammaPerSecond explorationC maxObsVel maxObsRadius", num, " ")
            for (i = 1; i <= n; i++) {
                # A column the CSV predates is a mismatch, not a pass.
                if (!(num[i] in h)) exit 1
                if (($h[num[i]] - w[i]) ^ 2 > 1e-12) exit 1
            }
            if (!("voGeometry" in h) || $h["voGeometry"] != w[6]) exit 1
            exit 0
        }
        END { if (NR < 2) exit 1 }   # header only, or empty
    ' "$1"
}

if [[ -f "${CSV}" ]]; then
    if params_match "${CSV}"; then
        echo "task $task_id: ${ALGO} ${SCENE} seed ${seed} already done"
        exit 0
    fi
    echo "task $task_id: ${CSV} exists but was produced with different" >&2
    echo "  parameters than this task's (rs=${RS} gamma=${GAMMA} c=${C}" >&2
    echo "  mov=${MOV}); re-running and overwriting it." >&2
fi

# ---------- ROS & environment ----------
set +eu
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -e

# Unique isolation per task
export ROS_DOMAIN_ID=$(( task_id % 101 ))
export ROS_LOCALHOST_ONLY=1
# Shared, deliberately. Here one task is a single run, so a per-task cache meant
# every one of the 360 runs recompiled the jitted kernels from scratch - about
# 15 s each. Every @jit in this project carries an explicit signature, so the
# kernels compile eagerly at import and the cache can be populated once, ahead
# of the campaign, by importing the modules (see docker/README.md).
#
# Default is /scratch, which is a bind mount and therefore shared by every node,
# so a single warm-up covers the whole campaign. Numba writes cache entries with
# atomic renames and keys them on the source path, which is identical across
# nodes, so concurrent use is safe. Set NUMBA_CACHE_DIR to /tmp/numba-shared to
# fall back to a node-local cache if the shared filesystem is ever a problem.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/numba-cache}"
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
