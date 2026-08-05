#!/usr/bin/env bash
#
# Parameter sweep over RADIUS_SCALE, the discount and the UCB exploration
# constant, several containers at a time.
#
#   docker/sweep.sh                     the default sweep, 3 containers
#   docker/sweep.sh --dry-run           print the configurations and stop
#   docker/sweep.sh -n 20 -j 3          20 runs per configuration
#   docker/sweep.sh --grid              every combination, not one axis at a time
#
# EACH CONFIGURATION GETS ITS OWN DIRECTORY under the sweep root:
#
#   sweeps/<date>-<time>/
#       configs.tsv                 what was run, one line per configuration
#       sweep.log                   the driver's own log
#       summary_all.csv             every configuration, one table
#       runs_all.csv                every run
#       rs1.80_g0.81_c1.00/         <- one directory per parameter set
#           config.env                  the exact arguments, machine-readable
#           debug/<scene>/              the run data: CSVs and pickles
#           logs/<scene>/               one console log per run
#           run.log                     the campaign's own output
#           summary.csv, runs.csv       this configuration, summarised
#
# The directory is what makes the parallelism safe. loopHandler_copy.py writes
# to a hardcoded debug/, and names its files by algorithm and run number only -
# not by the parameters - so three containers sharing one checkout would
# overwrite each other's results. Each container therefore gets its own
# debug/ and logs/ bind-mounted OVER the ones in the repo. The checkout itself,
# Unity builds included, stays shared and is never written to.
#
# Everything else that could collide is already isolated by the container:
# DDS discovery (own network namespace, so Unity and the planner only find
# their own partner), the Unity cleanup pkill in run_all_experiments.sh (own
# PID namespace, so it cannot kill another sweep's environment), and the numba
# cache (NUMBA_CACHE_DIR is inside the image, not the mount).
#
# READ THIS BEFORE COMPARING THE RESULTS TO ANYTHING ELSE
# -------------------------------------------------------
# Running N campaigns at once gives each of them a fraction of the machine, and
# the planner is given a TIME budget per step - so it gets through fewer
# simulations per step than it would alone. Configurations are therefore
# comparable WITH EACH OTHER, and not with a run made on an idle machine. Each
# container is pinned to its own fixed set of cores (--cpuset-cpus) so that the
# share is at least equal and constant across the sweep. simNum is recorded in
# every CSV: check it is roughly constant across configurations before reading
# anything into the results.
#
# Options:
#   -o, --out DIR            sweep root (default: sweeps/<date>-<time>)
#   -n, --num-exp N          runs per configuration (default: 20)
#   -j, --jobs N             containers in parallel (default: 3)
#   -a, --algorithms "A B"   default: "VO-TREE"
#   -t, --trajectories "X Y" default: "sinusoidal intention"
#       --radius-scale "..." values to sweep (default: "1.4 1.8 2.2 2.6")
#       --gamma "..."        values to sweep (default: "0.65 0.81 0.90 0.95")
#       --exploration-c "..." values to sweep (default: "0.5 1.0 2.0 5.0")
#       --grid               every combination instead of one axis at a time
#       --extra "ARGS"       extra arguments for every run
#       --dry-run            print what would run, then stop
#   -y, --yes                do not ask for confirmation
#   -h, --help               show this help and exit
set -euo pipefail

IMAGE="${IMAGE:-mctsvo:foxy}"
MOUNT=/ws/src/MCTS_VO_ROS

# Every path handed to "docker run -v" has to be absolute. Docker reads a
# source that does not start with / as the NAME of a named volume, and then
# rejects it for containing slashes:
#
#   create sweeps/.../debug: "sweeps/.../debug" includes invalid characters
#   for a local volume name
#
# "cd X && pwd" is the usual way to absolutise and it is not reliable here:
# with CDPATH set in the environment, cd echoes the directory it resolved to,
# so the command substitution captures that line as well as pwd's and the
# result is two paths joined by a newline. CDPATH= disables that, -P resolves
# symlinks, and -- stops a leading dash being read as an option.
abspath() {
    local p="$1" resolved
    [ -d "${p}" ] || mkdir -p "${p}"
    resolved="$(CDPATH= cd -P -- "${p}" && pwd)" || return 1
    printf '%s' "${resolved}"
}

# Belt and braces: if anything above ever fails to produce an absolute path,
# say which variable it was, rather than leaving docker to complain about a
# volume name several minutes into the sweep.
require_abs() {
    case "$2" in
        /*) ;;
        *) echo "sweep.sh: ${1} is not an absolute path: '${2}'" >&2
           echo "  This has to be absolute for 'docker run -v' to read it as a" >&2
           echo "  directory. Please report the line above." >&2
           exit 1 ;;
    esac
}

REPO="$(abspath "$(dirname "${BASH_SOURCE[0]}")/..")"
require_abs REPO "${REPO}"

# The B9 defaults. The centre point of the one-axis-at-a-time sweep, and what
# the two parameters not being swept are held at.
DEF_RS=1.8
DEF_GAMMA=0.81
DEF_C=1.0

NUM_EXP=20
JOBS=3
ALGORITHMS="VO-TREE"
TRAJECTORIES="sinusoidal intention"
RS_VALUES="1.4 1.8 2.2 2.6"
GAMMA_VALUES="0.65 0.81 0.90 0.95"
C_VALUES="0.5 1.0 2.0 5.0"
GRID=0
EXTRA=""
DRY_RUN=0
ASSUME_YES=0
OUT=""
# Rough, and only used for the estimate printed before the sweep starts: a
# 35 s episode at a 0.2 s control period is 70 s of driving, plus Unity's
# startup and the first run's numba compilation.
SECS_PER_RUN="${SECS_PER_RUN:-100}"

usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--out)            OUT="$2"; shift 2 ;;
        -n|--num-exp)        NUM_EXP="$2"; shift 2 ;;
        -j|--jobs)           JOBS="$2"; shift 2 ;;
        -a|--algorithms)     ALGORITHMS="$2"; shift 2 ;;
        -t|--trajectories)   TRAJECTORIES="$2"; shift 2 ;;
        --radius-scale)      RS_VALUES="$2"; shift 2 ;;
        --gamma)             GAMMA_VALUES="$2"; shift 2 ;;
        --exploration-c)     C_VALUES="$2"; shift 2 ;;
        --grid)              GRID=1; shift ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -y|--yes)            ASSUME_YES=1; shift ;;
        -h|--help)           usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[ -n "${OUT}" ] || OUT="${REPO}/sweeps/$(date +%Y%m%d-%H%M%S)"
OUT="$(abspath "${OUT}")"
require_abs OUT "${OUT}"
DRIVER_LOG="${OUT}/sweep.log"

# Identifies this sweep's containers, so Ctrl-C can stop exactly them and
# nothing else - another sweep running beside it, or an unrelated
# docker/run.sh, is left alone.
SWEEP_ID="$(basename "${OUT}")"
SWEEP_LABEL="mctsvo-sweep=${SWEEP_ID}"

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${DRIVER_LOG}"; }

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Image ${IMAGE} not found. Build it once with:" >&2
    echo "    docker/run.sh --build" >&2
    exit 1
fi

# The planner is a submodule, and a checkout does not populate it on its own.
# Without this the sweep would start, every single run would die on
# "No module named 'MCTS_VO.experiment_utils'", and the campaign script would
# report the failures and still exit 0 - so the sweep would look like it was
# working until it produced no data hours later. One second here instead.
if [ ! -f "${REPO}/mctsVoRos/MCTS_VO/experiment_utils.py" ]; then
    echo "The MCTS_VO submodule is not populated." >&2
    echo "    git submodule update --init" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# The configurations. A name that carries all three values, so a directory is
# self-describing and the naming survives a switch to --grid.
# ---------------------------------------------------------------------------
name_of() { printf 'rs%s_g%s_c%s' "$1" "$2" "$3"; }

CONFIGS=()      # "name<TAB>rs<TAB>gamma<TAB>c"
seen=""
add_config() {
    local n; n="$(name_of "$1" "$2" "$3")"
    case " ${seen} " in *" ${n} "*) return ;; esac    # the centre point of the
    seen="${seen} ${n}"                               # three axes coincides
    CONFIGS+=("${n}"$'\t'"$1"$'\t'"$2"$'\t'"$3")
}

if [ "${GRID}" -eq 1 ]; then
    for rs in ${RS_VALUES}; do
        for g in ${GAMMA_VALUES}; do
            for c in ${C_VALUES}; do add_config "${rs}" "${g}" "${c}"; done
        done
    done
else
    # One axis at a time: vary one parameter, hold the other two at the B9
    # default. The baseline is added first so it is the first thing to finish.
    add_config "${DEF_RS}" "${DEF_GAMMA}" "${DEF_C}"
    for rs in ${RS_VALUES}; do add_config "${rs}" "${DEF_GAMMA}" "${DEF_C}"; done
    for g  in ${GAMMA_VALUES}; do add_config "${DEF_RS}" "${g}" "${DEF_C}"; done
    for c  in ${C_VALUES}; do add_config "${DEF_RS}" "${DEF_GAMMA}" "${c}"; done
fi

n_algos=$(set -- ${ALGORITHMS};   echo $#)
n_scenes=$(set -- ${TRAJECTORIES}; echo $#)
RUNS_PER_CONFIG=$(( n_algos * n_scenes * NUM_EXP ))
TOTAL_RUNS=$(( ${#CONFIGS[@]} * RUNS_PER_CONFIG ))
EST_H=$(( TOTAL_RUNS * SECS_PER_RUN / JOBS / 3600 ))
EST_M=$(( TOTAL_RUNS * SECS_PER_RUN / JOBS % 3600 / 60 ))

# ---------------------------------------------------------------------------
# One fixed, disjoint set of cores per slot, so that every configuration gets
# the same share of the machine no matter what else in the sweep is running.
# ---------------------------------------------------------------------------
NCPU="$(nproc)"
CPUSETS=()
if [ "${NCPU}" -ge $(( JOBS * 2 )) ]; then
    per=$(( NCPU / JOBS )); extra=$(( NCPU % JOBS )); next=0
    for (( s = 0; s < JOBS; s++ )); do
        n=${per}; [ "${s}" -lt "${extra}" ] && n=$(( per + 1 ))
        CPUSETS+=( "${next}-$(( next + n - 1 ))" )
        next=$(( next + n ))
    done
else
    # Fewer than two cores each: pinning would starve Unity and the planner of
    # the same core. Let the scheduler decide, and say so.
    for (( s = 0; s < JOBS; s++ )); do CPUSETS+=( "" ); done
fi

{
    echo "=================================================================="
    echo " sweep root     ${OUT}"
    echo " image          ${IMAGE}"
    echo " configurations ${#CONFIGS[@]}"
    echo " algorithms     ${ALGORITHMS}"
    echo " scenes         ${TRAJECTORIES}"
    echo " runs each      ${NUM_EXP}  (${RUNS_PER_CONFIG} per configuration)"
    echo " total runs     ${TOTAL_RUNS}"
    echo " parallel       ${JOBS} containers on ${NCPU} cores: ${CPUSETS[*]:-unpinned}"
    echo " rough estimate ${EST_H}h ${EST_M}m at ${SECS_PER_RUN}s per run"
    echo "=================================================================="
} | tee -a "${DRIVER_LOG}"

printf 'name\tradius_scale\tgamma_per_second\texploration_c\n' > "${OUT}/configs.tsv"
for cfg in "${CONFIGS[@]}"; do printf '%s\n' "${cfg}" >> "${OUT}/configs.tsv"; done
column -t -s$'\t' "${OUT}/configs.tsv"

if [ "${DRY_RUN}" -eq 1 ]; then
    log "dry run, stopping here"
    exit 0
fi

if [ "${ASSUME_YES}" -eq 0 ]; then
    read -r -p "Start? [y/N] " reply
    case "${reply}" in [yY]*) ;; *) echo "aborted"; exit 1 ;; esac
fi

# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------
run_config() {
    local name="$1" rs="$2" g="$3" c="$4" slot="$5"
    local dir="${OUT}/${name}"

    # Created here, on the host, and not left to docker: a bind-mount source
    # that does not exist is created by the daemon as root, and then nothing
    # inside the container can write to it.
    mkdir -p "${dir}/debug" "${dir}/logs"
    require_abs "the directory for ${name}" "${dir}"

    cat > "${dir}/config.env" <<EOF
NAME=${name}
RADIUS_SCALE=${rs}
GAMMA_PER_SECOND=${g}
EXPLORATION_C=${c}
ALGORITHMS=${ALGORITHMS}
TRAJECTORIES=${TRAJECTORIES}
NUM_EXP=${NUM_EXP}
EXTRA=${EXTRA}
IMAGE=${IMAGE}
CPUSET=${CPUSETS[${slot}]}
STARTED=$(date -Iseconds)
EOF

    local args=(
        --rm --init
        # Labelled and named so that Ctrl-C, or anyone at all, can find these
        # containers again without having to guess from the image name.
        --label "${SWEEP_LABEL}"
        --name "mctsvo-sweep-${SWEEP_ID}-${name}"
        -v "${REPO}:${MOUNT}"
        -v "${dir}/debug:${MOUNT}/mctsVoRos/debug"
        -v "${dir}/logs:${MOUNT}/mctsVoRos/logs"
        -w "${MOUNT}/mctsVoRos"
        -e "ROS_DOMAIN_ID=$(( slot + 1 ))"
        --shm-size=1g
    )
    [ -n "${CPUSETS[${slot}]}" ] && args+=( --cpuset-cpus "${CPUSETS[${slot}]}" )

    # --env-render headless is not optional here: three windows would fight
    # over the display, and a windowed run is not comparable to a headless one
    # on the pre-fix builds anyway.
    docker run "${args[@]}" "${IMAGE}" \
        ./run_all_experiments.sh --skip-setup \
            -n "${NUM_EXP}" -a "${ALGORITHMS}" -t "${TRAJECTORIES}" \
            -x "--env-render headless --no-plots --radius-scale ${rs} --gamma-per-second ${g} --exploration-c ${c} ${EXTRA}" \
        >> "${dir}/run.log" 2>&1

    # run_all_experiments.sh prints how many runs failed and then exits 0
    # regardless, so its status says nothing about whether there is any data.
    # Count what actually landed instead.
    local produced expected
    produced="$(find "${dir}/debug" -name 'data_*.csv' 2>/dev/null | wc -l)"
    expected=$(( n_algos * n_scenes * NUM_EXP ))
    if [ "${produced}" -eq 0 ]; then
        {
            echo "NO RUN PRODUCED ANY DATA."
            echo "The first place to look is logs/<scene>/<algorithm>_0.log in"
            echo "this directory - the traceback will be at the end of it."
        } >> "${dir}/run.log"
        return 1
    fi
    if [ "${produced}" -lt "${expected}" ]; then
        echo "INCOMPLETE: ${produced} of ${expected} runs produced data." \
             >> "${dir}/run.log"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# The scheduler. A slot is a fixed set of cores, so a job has to wait for its
# own slot rather than for any free one.
# ---------------------------------------------------------------------------
declare -a SLOT_PID SLOT_NAME
for (( s = 0; s < JOBS; s++ )); do SLOT_PID[$s]=0; SLOT_NAME[$s]=""; done

reap() {
    local s=$1
    [ "${SLOT_PID[$s]}" -eq 0 ] && return 0
    if wait "${SLOT_PID[$s]}"; then
        log "done   ${SLOT_NAME[$s]}"
    else
        log "FAILED ${SLOT_NAME[$s]} (see ${OUT}/${SLOT_NAME[$s]}/run.log)"
        FAILURES=$(( FAILURES + 1 ))
    fi
    SLOT_PID[$s]=0
}

# Ctrl-C kills the driver and the docker CLIs it started, but a container whose
# CLI dies can be left running - and a stray Unity inside one will happily go on
# consuming a core until someone notices. Stop them by label, which is why they
# carry one. Interrupting is otherwise safe: every finished run is already on
# disk in its own directory, and rerunning with the same -o skips them.
stop_containers() {
    local ids
    ids="$(docker ps -q --filter "label=${SWEEP_LABEL}" 2>/dev/null)" || return 0
    [ -n "${ids}" ] || return 0
    echo
    echo "stopping $(echo "${ids}" | wc -l) container(s)..." >&2
    # shellcheck disable=SC2086
    docker stop ${ids} >/dev/null 2>&1 || true
}
on_interrupt() {
    trap - INT TERM
    echo >&2
    echo "Interrupted. Finished runs are kept; rerun with -o ${OUT} to resume." >&2
    stop_containers
    exit 130
}
trap on_interrupt INT TERM

FAILURES=0
START=$(date +%s)
DONE=0

for cfg in "${CONFIGS[@]}"; do
    IFS=$'\t' read -r name rs g c <<< "${cfg}"

    slot=-1
    while [ "${slot}" -lt 0 ]; do
        for (( s = 0; s < JOBS; s++ )); do
            if [ "${SLOT_PID[$s]}" -eq 0 ]; then slot=$s; break; fi
            if ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
                reap "$s"; DONE=$(( DONE + 1 )); slot=$s; break
            fi
        done
        [ "${slot}" -lt 0 ] && sleep 5
    done

    log "start  ${name}  [${DONE}/${#CONFIGS[@]} done, slot ${slot}, cores ${CPUSETS[${slot}]:-any}]"
    run_config "${name}" "${rs}" "${g}" "${c}" "${slot}" &
    SLOT_PID[$slot]=$!
    SLOT_NAME[$slot]="${name}"
done

for (( s = 0; s < JOBS; s++ )); do reap "$s"; done

ELAPSED=$(( $(date +%s) - START ))
log "all configurations finished in $(( ELAPSED / 3600 ))h $(( ELAPSED % 3600 / 60 ))m, ${FAILURES} failed"

# ---------------------------------------------------------------------------
# Summarising. One pass per configuration into its own directory, then one
# combined table with the parameters as leading columns.
# ---------------------------------------------------------------------------
log "summarising"
for cfg in "${CONFIGS[@]}"; do
    IFS=$'\t' read -r name rs g c <<< "${cfg}"
    docker run --rm -v "${REPO}:${MOUNT}" -v "${OUT}:/sweep" \
        -w "${MOUNT}/mctsVoRos" "${IMAGE}" \
        python3 summarize_debug.py --no-archive \
            --dir "/sweep/${name}/debug" \
            --csv "/sweep/${name}/summary.csv" \
            --runs-csv "/sweep/${name}/runs.csv" \
        >> "${OUT}/${name}/run.log" 2>&1 || log "summary failed for ${name}"
done

combine() {   # $1 = per-config file name, $2 = combined file name
    local first=1
    : > "${OUT}/$2"
    for cfg in "${CONFIGS[@]}"; do
        IFS=$'\t' read -r name rs g c <<< "${cfg}"
        local f="${OUT}/${name}/$1"
        [ -f "${f}" ] || continue
        awk -v n="${name}" -v rs="${rs}" -v g="${g}" -v c="${c}" -v first="${first}" '
            NR == 1 { if (first) print "config,radius_scale,gamma_per_second,exploration_c," $0; next }
            { print n "," rs "," g "," c "," $0 }' "${f}" >> "${OUT}/$2"
        first=0
    done
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
[ "${FAILURES}" -eq 0 ] || exit 1
