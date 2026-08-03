#!/usr/bin/env bash
#
# Runs the full experimental campaign of the paper:
#   3 algorithms x 30 runs x 2 obstacle-trajectory domains = 180 runs.
#
# The script is self-contained: it sources ROS, builds the workspace with
# colcon, sources the resulting overlay and activates the Python virtual
# environment before launching the experiments. Just run it:
#
#   ./run_all_experiments.sh
#
# It can be launched from any directory: paths are resolved relative to the
# location of the script itself.
#
# Results of each run are written by loopHandler_copy.py into
# debug/<trajectories>/ (e.g. debug/sinusoidal, debug/intention), animations into
# debug/<trajectories>/animations/, while the console output of every run is
# stored in logs/<trajectories>/.
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
#   -x, --extra "ARGS"       extra arguments passed straight to
#                            loopHandler_copy.py, e.g. to reproduce the
#                            published configuration:
#                              -x "--max-obs-vel 0.1 --exploration-c 10"
#       --skip-build         source and activate, but do not run colcon build
#       --skip-setup         run the experiments in the current shell, without
#                            sourcing/building/activating anything
#       --ros-distro NAME    ROS 2 distribution (default: $ROS_DISTRO or foxy)
#   -h, --help               show this help and exit

set -u

NUM_EXP=10
ALGORITHMS=("MCTS" "VO-TREE" "VO-PLANNER")
TRAJECTORIES=("sinusoidal" "intention")
FORCE=0
SKIP_BUILD=0
SKIP_SETUP=0
EXTRA_ARGS=()
ROS_DISTRO_NAME="${ROS_DISTRO:-foxy}"

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-exp)      NUM_EXP="$2"; shift 2 ;;
        -a|--algorithms)   read -r -a ALGORITHMS <<< "$2"; shift 2 ;;
        -t|--trajectories) read -r -a TRAJECTORIES <<< "$2"; shift 2 ;;
        -f|--force)        FORCE=1; shift ;;
        -x|--extra)        read -r -a EXTRA_ARGS <<< "$2"; shift 2 ;;
        --skip-build)      SKIP_BUILD=1; shift ;;
        --skip-setup)      SKIP_SETUP=1; shift ;;
        --ros-distro)      ROS_DISTRO_NAME="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths (resolved from the location of this script, so the working directory
# of the caller does not matter)
#   PKG_DIR  = .../MCTS_VO_ROS/mctsVoRos      experiments are run from here
#   PROJ_DIR = .../MCTS_VO_ROS                holds the virtual environment
#   WS_ROOT  = .../colcon_ws                  holds build/ install/ log/
# ---------------------------------------------------------------------------
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${PKG_DIR}/.." && pwd)"
WS_ROOT="$(cd "${PROJ_DIR}/../.." && pwd)"
VENV_DIR="${PROJ_DIR}/venv"

if [[ ! -f "${PKG_DIR}/loopHandler_copy.py" ]]; then
    echo "Error: loopHandler_copy.py not found in ${PKG_DIR}." >&2
    exit 1
fi

# ROS setup files and the venv activate script reference unbound variables,
# so `set -u` has to be relaxed while sourcing them.
source_quietly() {
    set +u
    # shellcheck disable=SC1090
    source "$1"
    local status=$?
    set -u
    return ${status}
}

setup_environment() {
    local ros_setup="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

    # 1. ROS 2 underlay
    if [[ ! -f "${ros_setup}" ]]; then
        echo "Error: ${ros_setup} not found." >&2
        echo "Set the right distribution with --ros-distro <name>." >&2
        exit 1
    fi
    echo ">> Sourcing ${ros_setup}"
    source_quietly "${ros_setup}"

    # 2. Build the workspace (with the venv still inactive, so that colcon
    #    builds against the system interpreter it was installed for)
    if [[ ${SKIP_BUILD} -eq 0 ]]; then
        echo ">> Building workspace in ${WS_ROOT}"
        if ! ( cd "${WS_ROOT}" && colcon build ); then
            echo "Error: colcon build failed, aborting." >&2
            exit 1
        fi
    else
        echo ">> Skipping colcon build"
    fi

    # 3. Workspace overlay
    local ws_setup="${WS_ROOT}/install/setup.bash"
    if [[ ! -f "${ws_setup}" ]]; then
        echo "Error: ${ws_setup} not found, build the workspace first." >&2
        exit 1
    fi
    echo ">> Sourcing ${ws_setup}"
    source_quietly "${ws_setup}"

    # 4. Python virtual environment (last, so its interpreter wins in PATH
    #    while the ROS packages stay reachable through PYTHONPATH)
    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        echo "Error: virtual environment not found in ${VENV_DIR}." >&2
        echo "Create it with: python3 -m venv --system-site-packages ${VENV_DIR}" >&2
        exit 1
    fi
    echo ">> Activating ${VENV_DIR}"
    source_quietly "${VENV_DIR}/bin/activate"

    # 5. Fail early rather than after a broken run
    if ! python3 -c "import rclpy, tf_transformations, numba" 2>/dev/null; then
        echo "Error: the Python environment is incomplete." >&2
        echo "Check that the venv was created with --system-site-packages and" >&2
        echo "that requirements.txt is installed." >&2
        exit 1
    fi
}

if [[ ${SKIP_SETUP} -eq 0 ]]; then
    setup_environment
else
    echo ">> Skipping environment setup"
fi

# Experiments use relative paths (../env_build/..., debug/...)
cd "${PKG_DIR}"

TOTAL=$(( ${#ALGORITHMS[@]} * ${#TRAJECTORIES[@]} * NUM_EXP ))
CURRENT=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

echo "=========================================================="
echo " Workspace:    ${WS_ROOT}"
echo " Algorithms:   ${ALGORITHMS[*]}"
echo " Trajectories: ${TRAJECTORIES[*]}"
echo " Runs each:    ${NUM_EXP}"
echo " Extra args:   ${EXTRA_ARGS[*]-<none>}"
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
                ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
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
echo " Results in: ${PKG_DIR}/debug"
echo "=========================================================="
