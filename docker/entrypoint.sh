#!/usr/bin/env bash
# Puts rclpy and the message packages on PYTHONPATH, then runs whatever was
# asked for. Without the source, "import rclpy" fails - the venv the host uses
# does not exist here, and does not need to: pip installed the requirements
# into the system interpreter, which is the one ROS itself uses.
set -e

# shellcheck disable=SC1091
source /opt/ros/foxy/setup.bash

# The repo is bind-mounted, so its files are not on the image's path. This lets
# "import mctsVoRos..." and the MCTS_VO submodule resolve from any working dir.
export PYTHONPATH="/ws/src/MCTS_VO_ROS/mctsVoRos:${PYTHONPATH}"

if [ ! -e /ws/src/MCTS_VO_ROS/mctsVoRos/loopHandler_copy.py ]; then
    echo "entrypoint: the repo is not mounted at /ws/src/MCTS_VO_ROS." >&2
    echo "  docker:  add -v <repo>:/ws/src/MCTS_VO_ROS, or use docker/run.sh" >&2
    echo "  EDF:     add the repo's parent to mounts = [...]" >&2
    exit 1
fi

exec "$@"
