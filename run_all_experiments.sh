#!/usr/bin/env bash
#
# Runs the full experimental campaign of the paper:
#   3 algorithms x 30 runs x 2 obstacle-trajectory domains = 180 runs.
#
# Results of each run are written by loopHandler_copy.py into
# debug/<trajectories>/ (e.g. debug/sinusoidal, debug/intention), animations into
# debug/<trajectories>/animations/, while the console output of every run is
# stored in logs/<trajectories>/.
#
# Must be executed from the mctsVoRos directory:
#   cd ~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos
#   ./run_all_experiments.sh
#
# Usage:
#   ./run_all_experiments.sh [options]
#
# Options:
#   -n, --num-exp N          number of runs per configuration (default: 30)
#   -a, --algorithms "A B"   space separated list of algorithms
#                            (default: "MCTS VO-TREE VO-PLANNER")
#   -t, --trajectories "X Y" space separated list of trajectory types
#                            (default: "sinusoidal intention")
#   -f, --force              re-run configurations that already have results
#                            (by default completed runs are skipped, so the
#                             campaign can be resumed after an interruption)
#   -h, --help               show this help and exit

set -u

NUM_EXP=30
ALGORITHMS=("MCTS" "VO-TREE" "VO-PLANNER")
TRAJECTORIES=("sinusoidal" "intention")
FORCE=0

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-exp)
            NUM_EXP="$2"; shift 2 ;;
        -a|--algorithms)
            read -r -a ALGORITHMS <<< "$2"; shift 2 ;;
        -t|--trajectories)
            read -r -a TRAJECTORIES <<< "$2"; shift 2 ;;
        -f|--force)
            FORCE=1; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# Sanity check: the script relies on relative paths, so it has to be launched
# from the mctsVoRos directory.
if [[ ! -f "loopHandler_copy.py" ]]; then
    echo "Error: loopHandler_copy.py not found." >&2
    echo "Run this script from the mctsVoRos directory." >&2
    exit 1
fi

TOTAL=$(( ${#ALGORITHMS[@]} * ${#TRAJECTORIES[@]} * NUM_EXP ))
CURRENT=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

echo "=========================================================="
echo " Algorithms:   ${ALGORITHMS[*]}"
echo " Trajectories: ${TRAJECTORIES[*]}"
echo " Runs each:    ${NUM_EXP}"
echo " Total runs:   ${TOTAL}"
echo "=========================================================="

# Make sure no leftover Unity instance is running before starting, and clean up
# on exit (e.g. when the user presses Ctrl-C).
cleanup() {
    pkill -f "env_build/.*env.x86_64" 2>/dev/null
}
trap 'echo; echo "Interrupted, stopping."; cleanup; exit 130' INT TERM
cleanup

for traj in "${TRAJECTORIES[@]}"; do
    mkdir -p "debug/${traj}" "logs/${traj}"

    for algo in "${ALGORITHMS[@]}"; do
        for (( exp=0; exp<NUM_EXP; exp++ )); do
            CURRENT=$(( CURRENT + 1 ))
            result_file="debug/${traj}/data_${algo}_${exp}.csv"
            log_file="logs/${traj}/${algo}_${exp}.log"

            if [[ ${FORCE} -eq 0 && -f "${result_file}" ]]; then
                echo "[${CURRENT}/${TOTAL}] skip ${traj} | ${algo} | run ${exp} (already done)"
                SKIPPED=$(( SKIPPED + 1 ))
                continue
            fi

            echo "[${CURRENT}/${TOTAL}] run  ${traj} | ${algo} | run ${exp}"

            python3 loopHandler_copy.py \
                --exp_num "${exp}" \
                --algorithm "${algo}" \
                --trajectories "${traj}" \
                > "${log_file}" 2>&1

            status=$?
            if [[ ${status} -ne 0 ]]; then
                echo "    warning: exited with status ${status}, see ${log_file}"
                FAILED=$(( FAILED + 1 ))
            fi

            # The Unity build is spawned by the Python script; make sure it is
            # really gone before starting the next run.
            cleanup
            sleep 2
        done
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "=========================================================="
echo " Done in $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
echo " Executed: $(( CURRENT - SKIPPED ))   Skipped: ${SKIPPED}   Failed: ${FAILED}"
echo " Results in: debug/{$(IFS=,; echo "${TRAJECTORIES[*]}")}"
echo "=========================================================="