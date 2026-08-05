#!/usr/bin/env bash
#
# Run the whole sweep, JOBS tasks at a time, then summarise it.
#
#   ./docker/sweep.sh              the lot
#   JOBS=2 ./docker/sweep.sh       two at a time instead of three
#   NUM_EXP=1 ./docker/sweep.sh    a quick smoke test
#
# One task is one run: one parameter set, one scene, one run number.
# docker/sweep_run.sh holds the parameter values and does the work.
set -e

JOBS=${JOBS:-3}

SWEEP_DIR=${SWEEP_DIR:-"$PWD/sweeps/$(date +%F_%H-%M-%S)"}
export SWEEP_DIR
mkdir -p "$SWEEP_DIR"

./docker/sweep_run.sh --tsv > "$SWEEP_DIR/configs.tsv"
n=$(./docker/sweep_run.sh --count)

echo "$n tasks, $JOBS at a time, into $SWEEP_DIR"

# Ctrl-C on its own is NOT enough. xargs does not pass the signal on to the
# tasks it has started, and a `docker run` without a terminal does not forward
# it to the container either - so the containers, and the Unity inside them,
# carry on. Measured: three containers still running, and new tasks still
# starting, ten seconds after the interrupt.
#
# Stopping the containers is what actually unwinds it: each docker run then
# returns, its sweep_run.sh exits, and xargs finishes. Killing the children
# afterwards stops xargs handing out any more ids.
#
# Interrupting costs only the runs in flight. Every finished run is already
# written to its parameter set's directory.
stop_everything() {
    trap - INT TERM
    echo >&2
    echo "stopping..." >&2
    local ids
    ids="$(docker ps -q --filter label=mctsvo-sweep)" || true
    # shellcheck disable=SC2086
    [ -n "$ids" ] && docker stop $ids >/dev/null 2>&1
    pkill -P $$ >/dev/null 2>&1 || true
    echo "stopped. Finished runs are in $SWEEP_DIR" >&2
    exit 130
}
trap stop_everything INT TERM

# -n 1: one task id per invocation. -P: that many at once, starting the next
# as soon as one finishes. Nothing is shared between them but the checkout.
# || true so that a failed task does not stop the summary from running; each
# failure says on stderr which log to look in.
seq 0 $(( n - 1 )) | xargs -P "$JOBS" -n 1 ./docker/sweep_run.sh || true

./docker/summarize_sweep.sh "$SWEEP_DIR"