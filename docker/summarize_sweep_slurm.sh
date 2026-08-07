#!/usr/bin/env bash
#
# Summarise a sweep on the cluster. Same job as docker/summarize_sweep.sh, but
# it calls summarize_debug.py directly instead of through "docker run" - there
# is no docker daemon on a compute node.
#
# Run it INSIDE the container:
#
#   srun -A <account> --partition=normal --time=00:30:00 \
#        --environment=mctsvo /workspace/MCTS_VO_ROS/docker/summarize_sweep_slurm.sh
#
# Reads every parameter set directly off the directory names, so it does not
# need configs.tsv and does not care whether the sweep finished.
set -e

SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"
REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

# summarize_debug.py imports the project, which needs ROS on the path. The
# container engine does not run the image's ENTRYPOINT, which is where that
# normally happens.
set +eu
# shellcheck disable=SC1091
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -e
export HOME="${HOME_OVERRIDE:-/tmp/summarize-home}"
export MPLCONFIGDIR="$HOME/.mpl"
mkdir -p "$MPLCONFIGDIR"

cd "$REPO/mctsVoRos"

n=0
for d in "$SWEEP_DIR"/rs*/; do
    [ -d "$d/debug" ] || continue
    name="$(basename "$d")"
    runs=$(find "$d/debug" -name 'data_*.csv' | wc -l)
    if [ "$runs" -eq 0 ]; then
        echo "skip $name (no runs)"
        continue
    fi
    python3 summarize_debug.py --no-archive \
        --dir "$d/debug" \
        --csv "$d/summary.csv" \
        --runs-csv "$d/runs.csv" > "$d/summary.txt" 2>&1 \
        || { echo "FAILED $name - see $d/summary.txt"; continue; }
    echo "$name  ($runs runs)"
    n=$(( n + 1 ))
done

echo "summarised $n parameter sets"

# --------------------------------------------------
# Combine, with the parameters as leading columns. The values come out of the
# directory name, which is the only place they are recorded for certain.
# --------------------------------------------------
combine() {   # $1 = per-set file, $2 = combined file
    local first=1 f name rs g c vo
    : > "$SWEEP_DIR/$2"
    for d in "$SWEEP_DIR"/rs*/; do
        f="$d/$1"
        [ -f "$f" ] || continue
        name="$(basename "$d")"
        # Every field is taken up to the next underscore, never to the end of
        # the name: the name gained a _vo suffix, and "${name##*_c}" would have
        # swallowed it into c.
        rs="${name#rs}"; rs="${rs%%_*}"
        g="${name#*_g}"; g="${g%%_*}"
        c="${name#*_c}"; c="${c%%_*}"
        # Sweeps predating the VO A/B have no _vo suffix; they are all the old
        # geometry, but say so rather than guessing.
        case "$name" in
            *_vo*) vo="${name##*_vo}" ;;
            *)     vo="unknown" ;;
        esac
        awk -v n="$name" -v rs="$rs" -v g="$g" -v c="$c" -v vo="$vo" -v first="$first" '
            NR == 1 { if (first) print "config,radius_scale,gamma_per_second,exploration_c,vo_geometry," $0; next }
            { print n "," rs "," g "," c "," vo "," $0 }' "$f" >> "$SWEEP_DIR/$2"
        first=0
    done
}

combine summary.csv summary_all.csv
combine runs.csv    runs_all.csv

echo
echo "$SWEEP_DIR/summary_all.csv   one row per parameter set per scene"
echo "$SWEEP_DIR/runs_all.csv      one row per run"
echo
echo "REMEMBER: discountedReturn is NOT comparable across the gamma axis."
echo "Read undiscReturn, goalPct and the collision columns there."
echo "Check sims is roughly constant across sets before trusting differences."
