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

# Per-scene parameters. The search freezes obstacles at their observed position
# for the whole rollout (paper 4.2.1), so the tuning is really about how far out
# that fiction is still trusted - which is what gamma sets, through
# horizon = dt/(1 - gamma^dt) and the derived DEPTH.
#
# PROVENANCE. GAMMA comes from a 6400-run sweep (8 gamma values x 4 scenes x
# 200 seeds), then RS and MOV from a 12000-run sweep (6 rs x 5 mov x 4 scenes
# x 100 seeds, gamma held at the first sweep's per-scene value), both run on
# the 583611d scenes (post-40b4070/583611d redesign). Epoch 4. This overturned
# every value from the pre-redesign 358,400-run sweep; none of that history
# still applies and is not repeated here.
#
# rs and mov point in OPPOSITE directions on different scenes. On intention,
# larger rs/mov monotonically helps, peaking at the top of the tested range
# (rs=2.0, mov=0.25, 90% goal, still rising there - extend the grid upward
# before trusting this as a true optimum). sinusoidal, intention_complex and
# sinusoidal_complex all peak at rs=1.0, the BOTTOM of the tested range
# (intention_complex is still falling there too - extend downward). Mechanism:
# the crowded scenes have obstacles down to 0.2m apart (583611d); a large
# rs/mov cone from one obstacle overlaps its neighbour's, so A_c=Ø (the
# Algorithm-4 trapped fallback) triggers far more often and obsColl climbs
# past 90%. A large cone only helps when obstacles are isolated, which is only
# true on intention.
#
# MOV must be >= the true per-scene obstacle speed or the VO guarantee is
# void - not a tuning knob to shrink for a better score. The true maxima,
# read off the serialised scene values (Scenes/*.unity, move_1.cs/move_2.cs,
# confirmed independently of the ef9eb8f commit message's 0.1/0.2 claim,
# which describes a DIFFERENT mover pair sharing the same scenes):
#   easy scenes (INT_EASY, SIN_EASY):       move_1/move_2 maxSpeed = 0.15
#   complex scenes (INT_COMPLEX, SIN_COMPLEX): move_1/move_2 maxSpeed = 0.20
# Three of the four raw sweep-best cells had mov BELOW this (sinusoidal at
# 0.10, intention_complex at 0.15, sinusoidal_complex at 0.125) - their good
# scores partly reflect an undersized VO ball, not a better plan.
#
# Above the floor, MOV was picked per scene by whether the extra margin buys
# real safety at n=100/cell, not just goal%:
#   intention: floor (0.15) already has volColl=0 across every tested mov, so
#     the raw-best 0.25 buys nothing measurable and isn't worth an unverified
#     margin - use the floor.
#   sinusoidal: floor (0.15) IS the raw-best cell (97% goal, volColl=0) -
#     nothing to trade off.
#   intention_complex: floor (0.20) gives 70% goal/5% volColl; 0.25 gives 60%/
#     1%. At n=100 a 5% and a 1% rate have overlapping 95% CIs (roughly
#     0.7-9.3% vs 0-3.9%) while the 10-point goal drop is not noise - paying
#     for an unproven safety gain isn't justified, use the floor.
#   sinusoidal_complex: floor (0.20) gives 8%/5%; 0.25 gives 7%/0% - a 1-point
#     goal cost for volColl clearly to zero. Worth it - use 0.25, not the
#     floor.
#
# c is flat everywhere except sinusoidal, which wants >= 2.0; 5.0 unchanged.
case "$SCENE" in
    sinusoidal)          RS=1.0; GAMMA=0.50; C=5.0;  MOV=0.15  ;;
    sinusoidal_complex)  RS=1.0; GAMMA=0.22; C=5.0;  MOV=0.25  ;;
    intention)           RS=2.0; GAMMA=0.03; C=5.0;  MOV=0.15  ;;
    intention_complex)   RS=1.0; GAMMA=0.03; C=5.0;  MOV=0.20  ;;
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

# Resume, but only over a run made with THESE parameters. A bare [[ -f ]] once
# let a c=5.0 campaign inherit 26 of 30 seeds from a c=1.0 run, and listing
# parameters alone was not enough either: when DEPTH became a function of gamma
# every listed value still matched and a whole campaign skipped itself. Hence
# also PARAM_EPOCH, read from the source so the two cannot drift.
PARAM_EPOCH="$(grep -oE '^PARAM_EPOCH = [0-9]+' \
    "${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}/mctsVoRos/loopHandler_copy.py" \
    | grep -oE '[0-9]+$')"
if [[ -z "$PARAM_EPOCH" ]]; then
    echo "could not read PARAM_EPOCH from loopHandler_copy.py" >&2
    exit 1
fi

params_match() {
    awk -F, -v want="$RS|$GAMMA|$C|$MOV|${MAX_OBS_RADIUS:-0.5}|${VO_GEOMETRY:-paper}|$PARAM_EPOCH|${RANGE_METRIC:-norm}" '
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
            # Runs predating the stamp have no epoch column and are therefore
            # from epoch 1 or earlier: never reusable once it has been bumped.
            if (!("paramEpoch" in h) || ($h["paramEpoch"] + 0) != (w[7] + 0)) exit 1
            if (!("rangeMetric" in h) || $h["rangeMetric"] != w[8]) exit 1
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
# Shared on purpose: one task is one run, so a per-task cache recompiled the
# jitted kernels every time (~15 s). /scratch is a bind mount, so one warm-up
# covers the campaign; numba writes with atomic renames, so this is concurrency
# safe. Use /tmp/numba-shared to fall back to node-local.
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

# Isolated working directory, mirroring the sweep layout: env_build one level
# above the work directory, because loopHandler uses the relative ../env_build.
CELL_DIR="/tmp/campaign-cell-${task_id}"
WORK="${CELL_DIR}/work"
mkdir -p "$WORK"

REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

ln -sfn "$REPO/env_build" "$CELL_DIR/env_build"

# Symlink the project code into the work directory
ln -sfn "$REPO/mctsVoRos/MCTS_VO" "$WORK/MCTS_VO"
for f in "$REPO"/mctsVoRos/*.py; do
    ln -sfn "$f" "$WORK/$(basename "$f")"
done

# Point the local debug/ to the shared repository debug folder
ln -sfn "$DEBUG_ROOT" "$WORK/debug"

cd "$WORK"

# Stagger the eight procs so they do not all JIT into a cold cache at once.
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
        --range-metric "${RANGE_METRIC:-norm}" \
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
