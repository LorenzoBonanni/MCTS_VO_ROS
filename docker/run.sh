#!/usr/bin/env bash
#
# Run anything from this repo inside the container.
#
#   docker/run.sh                                   interactive shell
#   docker/run.sh python3 loopHandler_copy.py --algorithm VO-TREE \
#                 --trajectories intention --exp_num 0
#   docker/run.sh ./run_all_experiments.sh -n 30
#   docker/run.sh python3 summarize_debug.py --no-csv
#
#   docker/run.sh --build       rebuild the image, then carry on
#   docker/run.sh --net-host    share the host network stack (see README)
#
# The working directory inside the container is mctsVoRos/, the same place the
# commands are run from on the host, so every command that works there works
# here unchanged.
set -euo pipefail

IMAGE="${IMAGE:-mctsvo:foxy}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT=/ws/src/MCTS_VO_ROS

BUILD=0
NET_HOST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build)    BUILD=1; shift ;;
        --net-host) NET_HOST=1; shift ;;
        --)         shift; break ;;
        *)          break ;;
    esac
done

if [ "${BUILD}" -eq 1 ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo ">>> building ${IMAGE} (a few minutes the first time)"
    # video/render group ids differ between hosts and must match for /dev/dri
    # to be usable from inside the container.
    # id -g, not the effective gid: under "sg docker" the effective primary
    # group is docker, and baking that in makes every output file group-docker.
    docker build \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g "$(id -un)")" \
        --build-arg VIDEO_GID="$(getent group video  | cut -d: -f3)" \
        --build-arg RENDER_GID="$(getent group render | cut -d: -f3)" \
        -f "${REPO}/docker/Dockerfile" \
        -t "${IMAGE}" \
        "${REPO}"
fi

ARGS=(
    --rm
    --init                          # reap the Unity process if a run is killed
    -v "${REPO}:${MOUNT}"
    -w "${MOUNT}/mctsVoRos"
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
    # Unity and the planner discover each other over DDS inside this container.
    # Both ends bundle their own RTPS stack, so nothing is exchanged with the
    # host and two containers cannot interfere with each other.
    --shm-size=1g
)

# Display passthrough, so --env-render window works. Harmless when the run is
# headless: nothing opens the display. Only set up when there is an X socket to
# hand, which there is not over ssh without forwarding.
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    ARGS+=( -e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:ro )
    [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ] && \
        ARGS+=( -e XAUTHORITY=/tmp/.docker.xauth -v "${XAUTHORITY}:/tmp/.docker.xauth:ro" )
    # Hardware rendering. Without it Unity falls back to llvmpipe, which still
    # runs - obstacle speed no longer depends on the frame rate - but slowly.
    [ -d /dev/dri ] && ARGS+=( --device /dev/dri )
else
    echo ">>> no DISPLAY: headless only (pass --env-render headless)" >&2
fi

[ "${NET_HOST}" -eq 1 ] && ARGS+=( --network host )
[ -t 0 ] && ARGS+=( -it )

exec docker run "${ARGS[@]}" "${IMAGE}" "${@:-bash}"
