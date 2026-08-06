#!/usr/bin/env bash
#
# Run ONE experiment: one parameter set, one scene, one run number. The task id
# picks all of them, the way a Slurm array index does. No run_all_experiments.sh:
# this calls loopHandler_copy.py directly.
#
#   ./docker/sweep_run.sh <task_id>     run one task
#   ./docker/sweep_run.sh --count       how many tasks there are
#   ./docker/sweep_run.sh --tsv         the parameter sets, for summarize_sweep
#   ./docker/sweep_run.sh --list        the grid and its size
#
# docker/sweep.sh runs all of them and summarises. To do it by hand:
#
#   export SWEEP_DIR="$PWD/sweeps/$(date +%F_%H-%M-%S)"
#   seq 0 2559 | xargs -P 3 -n 1 ./docker/sweep_run.sh
#
# SWEEP_DIR has to be exported, not computed here: every task must land in the
# same directory, and each task is a separate process.
set -e

IMAGE=${IMAGE:-mctsvo:foxy}
ALGORITHM=${ALGORITHM:-VO-TREE}

# --------------------------------------------------
# 1. The grid: rs x gamma x c x scene x run
# --------------------------------------------------
# THE FULL PRODUCT. Every value of every list is combined with every value of
# every other, so the count grows fast - check --list before launching. The
# defaults (1.8, 0.81, 1.0) are in the lists, so the baseline is one of the
# cells and needs no special case.
rs_values=(1.4 1.8 2.2 2.6)
gamma_values=(0.65 0.81 0.90 0.95)
c_values=(0.5 1.0 2.0 5.0)
scenes=(sinusoidal intention)
num_runs=${NUM_EXP:-100}

num_rs=${#rs_values[@]}
num_gamma=${#gamma_values[@]}
num_c=${#c_values[@]}
num_scenes=${#scenes[@]}
total_tasks=$(( num_rs * num_gamma * num_c * num_scenes * num_runs ))

name_of() { echo "rs${1}_g${2}_c${3}"; }

case "${1:-}" in
    --count)
        echo "$total_tasks"; exit 0 ;;
    --tsv)
        # One line per PARAMETER SET, not per task: the scene is a directory
        # inside each set's debug/, which is what summarize_sweep.sh expects.
        printf 'name\trs\tgamma\tc\n'
        for rs in "${rs_values[@]}"; do
          for g in "${gamma_values[@]}"; do
            for c in "${c_values[@]}"; do
                printf '%s\t%s\t%s\t%s\n' "$(name_of "$rs" "$g" "$c")" "$rs" "$g" "$c"
            done
          done
        done
        exit 0 ;;
    --list|"")
        echo "rs      ${rs_values[*]}    ($num_rs)"
        echo "gamma   ${gamma_values[*]}    ($num_gamma)"
        echo "c       ${c_values[*]}    ($num_c)"
        echo "scenes  ${scenes[*]}    ($num_scenes)"
        echo "runs    $num_runs"
        echo
        echo "$(( num_rs * num_gamma * num_c )) parameter sets"
        echo "$total_tasks tasks, ids 0 to $(( total_tasks - 1 ))"
        echo "roughly $(( total_tasks * 90 / 3600 )) core-hours at 90 s a run"
        exit 0 ;;
esac

task_id=$1
if (( task_id < 0 || task_id >= total_tasks )); then
    echo "task_id $task_id out of range 0..$(( total_tasks - 1 ))" >&2
    exit 1
fi

rs_idx=$((   task_id / (num_gamma * num_c * num_scenes * num_runs) ))
gamma_idx=$(( (task_id / (num_c * num_scenes * num_runs)) % num_gamma ))
c_idx=$((     (task_id / (num_scenes * num_runs)) % num_c ))
scene_idx=$(( (task_id / num_runs) % num_scenes ))
exp_num=$((   task_id % num_runs ))

RS="${rs_values[$rs_idx]}"
GAMMA="${gamma_values[$gamma_idx]}"
C="${c_values[$c_idx]}"
SCENE="${scenes[$scene_idx]}"
NAME="$(name_of "$RS" "$GAMMA" "$C")"

# --------------------------------------------------
# 2. Checks
# --------------------------------------------------
# Absolute: docker reads a -v source that does not start with / as the name of
# a named volume, and rejects it for containing slashes.
: "${SWEEP_DIR:?export SWEEP_DIR to an absolute path first}"
[[ "$SWEEP_DIR" == /* ]] || { echo "SWEEP_DIR must be absolute" >&2; exit 1; }

[[ -f mctsVoRos/loopHandler_copy.py ]] || {
    echo "run this from the repository root" >&2; exit 1; }

# Without the submodule loopHandler dies on import, immediately, every time.
[[ -f mctsVoRos/MCTS_VO/experiment_utils.py ]] || {
    echo "submodule not populated: git submodule update --init" >&2; exit 1; }

# --------------------------------------------------
# 3. Run it
# --------------------------------------------------
# One debug/ per parameter set, mounted over the repository's. loopHandler
# writes to a hardcoded debug/<scene>/ and names files data_<algorithm>_<run>,
# never by the parameters - so two parameter sets sharing a checkout would
# overwrite each other. Two tasks of the SAME set are fine side by side: a
# different scene is a different subdirectory, a different run number is a
# different filename.
DIR="$SWEEP_DIR/$NAME"
mkdir -p "$DIR/debug" "$DIR/logs/$SCENE"

cat > "$DIR/config.env" <<EOF
RADIUS_SCALE=$RS
GAMMA=$GAMMA
EXPLORATION_C=$C
EOF

echo "task $task_id: $NAME | $SCENE | run $exp_num"

# Its own docker run rather than docker/run.sh: the sweep needs one extra -v,
# and run.sh builds its argument list internally. Everything else matches
# run.sh - same image, same mount point, same working directory. Headless, so
# none of run.sh's display passthrough applies.
docker run --rm --init \
    --label mctsvo-sweep \
    --shm-size=1g \
    -v "$PWD:/ws/src/MCTS_VO_ROS" \
    -v "$DIR/debug:/ws/src/MCTS_VO_ROS/mctsVoRos/debug" \
    -w /ws/src/MCTS_VO_ROS/mctsVoRos \
    "$IMAGE" \
    python3 loopHandler_copy.py \
        --algorithm "$ALGORITHM" \
        --trajectories "$SCENE" \
        --exp_num "$exp_num" \
        --env-render headless \
        --no-plots \
        --radius-scale "$RS" \
        --gamma-per-second "$GAMMA" \
        --exploration-c "$C" \
    > "$DIR/logs/$SCENE/${ALGORITHM}_${exp_num}.log" 2>&1

# loopHandler finishes by raising and catching an exception, so a normal end
# and a crash look alike from outside. The CSV is the real signal.
if [[ ! -f "$DIR/debug/$SCENE/data_${ALGORITHM}_${exp_num}.csv" ]]; then
    echo "task $task_id: NO DATA - see $DIR/logs/$SCENE/${ALGORITHM}_${exp_num}.log" >&2
    exit 1
fi

echo "task $task_id: done"