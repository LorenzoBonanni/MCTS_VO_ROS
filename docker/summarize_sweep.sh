#!/usr/bin/env bash
#
# Summarise a sweep directory: one summary.csv/runs.csv per configuration,
# then a combined summary_all.csv/runs_all.csv with the parameters as leading
# columns. Reads configs.tsv, so it needs nothing from the sweep run except
# the directory it left behind - call it standalone against any sweep root,
# or let sweep.sh call it itself when a campaign finishes.
#
#   docker/summarize_sweep.sh sweeps/20260101-120000
set -euo pipefail

IMAGE="${IMAGE:-mctsvo:foxy}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOUNT=/ws/src/MCTS_VO_ROS

[ $# -eq 1 ] || { echo "usage: $0 <sweep-dir>" >&2; exit 1; }
OUT="$(cd "$1" && pwd)"
CONFIGS_TSV="${OUT}/configs.tsv"
[ -f "${CONFIGS_TSV}" ] || { echo "no configs.tsv in ${OUT}" >&2; exit 1; }

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${OUT}/sweep.log"; }

# ---------------------------------------------------------------------------
# Summarising. One pass per configuration into its own directory, then one
# combined table with the parameters as leading columns.
# ---------------------------------------------------------------------------
log "summarising"
SUM_FAILURES=0
while IFS=$'\t' read -r name rs g c; do
    docker run --rm -v "${REPO}:${MOUNT}" -v "${OUT}:/sweep" \
        -w "${MOUNT}/mctsVoRos" "${IMAGE}" \
        python3 summarize_debug.py --no-archive \
            --dir "/sweep/${name}/debug" \
            --csv "/sweep/${name}/summary.csv" \
            --runs-csv "/sweep/${name}/runs.csv" \
        >> "${OUT}/${name}/run.log" 2>&1 || { log "summary failed for ${name}"; SUM_FAILURES=$(( SUM_FAILURES + 1 )); }
done < <(tail -n +2 "${CONFIGS_TSV}")

combine() {   # $1 = per-config file name, $2 = combined file name
    local first=1
    : > "${OUT}/$2"
    while IFS=$'\t' read -r name rs g c; do
        local f="${OUT}/${name}/$1"
        [ -f "${f}" ] || continue
        awk -v n="${name}" -v rs="${rs}" -v g="${g}" -v c="${c}" -v first="${first}" '
            NR == 1 { if (first) print "config,radius_scale,gamma_per_second,exploration_c," $0; next }
            { print n "," rs "," g "," c "," $0 }' "${f}" >> "${OUT}/$2"
        first=0
    done < <(tail -n +2 "${CONFIGS_TSV}")
}
combine summary.csv summary_all.csv
combine runs.csv    runs_all.csv

log "wrote ${OUT}/summary_all.csv and ${OUT}/runs_all.csv"
echo
echo "Per configuration:  ${OUT}/<name>/"
echo "Everything at once: ${OUT}/summary_all.csv"
echo
echo "REMEMBER: discountedReturn is not comparable across a change of --gamma."
echo "Read undiscountedReturn, the success rate and the collision counts on the"
echo "gamma axis. Check simNum is roughly constant before trusting any of it."
[ "${SUM_FAILURES}" -eq 0 ] || exit 1
