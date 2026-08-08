#!/usr/bin/env bash
#
# One cell of the sweep: one (rs, gamma, c, scene), every run number for it.
# Launched by sweep.sbatch through srun, one per task on the node.
#
# Every parameter comes from sweep.sbatch's exports; this script defines no
# defaults and refuses to start without them. To run a single cell by hand,
# source those exports first:
#
#   set -a; . <(grep '^export ' docker/sweep.sbatch); set +a
#   SLURM_ARRAY_TASK_ID=0 SLURM_PROCID=0 ./sweep_worker.sh
set -e

# --------------------------------------------------
# 0. Task id
# --------------------------------------------------
if [[ -z "${SLURM_ARRAY_TASK_ID}" || -z "${SLURM_PROCID}" ]]; then
    echo "expects SLURM_ARRAY_TASK_ID and SLURM_PROCID to be set" >&2
    exit 1
fi
task_id=$(( SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID ))

# Every one of these is set by sweep.sbatch. They are checked up front, and with
# no defaults: a missing grid variable used to produce a zero-length array, hence
# num_* == 0 and a division by zero forty lines below, which says nothing about
# what is actually wrong.
: "${SWEEP_DIR:?export SWEEP_DIR}"
: "${MCTSVO_REPO:?export MCTSVO_REPO}"
: "${RS_VALS:?export RS_VALS}"
: "${GAMMA_VALS:?export GAMMA_VALS}"
: "${C_VALS:?export C_VALS}"
: "${TRAJECTORIES:?export TRAJECTORIES}"
: "${VO_GEOMETRIES:?export VO_GEOMETRIES}"
: "${MAX_OBS_VEL_VALS:?export MAX_OBS_VEL_VALS}"
: "${ALGORITHM:?export ALGORITHM}"
: "${NUM_EXP:?export NUM_EXP}"
: "${MAX_OBS_RADIUS:?export MAX_OBS_RADIUS}"
REPO="$MCTSVO_REPO"

# --------------------------------------------------
# 1. The grid. ONE CELL PER TASK: the run number is NOT part of the index, the
#    loop at the bottom covers it.
#
#    Every axis comes from sweep.sbatch. These lists used to be hard-coded here
#    while sweep.sbatch exported RS_VALS/GAMMA_VALS/C_VALS/TRAJECTORIES that
#    nothing read, so editing the sbatch changed nothing and the two files could
#    disagree silently - which they did: the sbatch said "sinusoidal intention"
#    while every sweep actually ran on the _complex scenes.
# --------------------------------------------------
IFS=' ' read -ra rs_values    <<< "$RS_VALS"
IFS=' ' read -ra gamma_values <<< "$GAMMA_VALS"
IFS=' ' read -ra c_values     <<< "$C_VALS"
IFS=' ' read -ra scenes       <<< "$TRAJECTORIES"
IFS=' ' read -ra mov_values   <<< "$MAX_OBS_VEL_VALS"
IFS=' ' read -ra vo_values    <<< "$VO_GEOMETRIES"

num_rs=${#rs_values[@]}
num_gamma=${#gamma_values[@]}
num_c=${#c_values[@]}
num_scenes=${#scenes[@]}
num_mov=${#mov_values[@]}
num_vo=${#vo_values[@]}
total_cells=$(( num_rs * num_gamma * num_c * num_scenes * num_mov * num_vo ))

if (( task_id >= total_cells )); then
    echo "task_id $task_id beyond the grid (0..$(( total_cells - 1 ))), nothing to do"
    exit 0
fi

# configs.tsv is the grid manifest. Nothing reads it any more -
# summarize_sweep_slurm.sh recovers the parameters from the directory names -
# but it is the one place the intended grid is recorded, which matters when a
# sweep is only partly finished and the directories are therefore incomplete.
# The first task of the first array job writes it, once: any other task would
# be racing, and every task has the same lists anyway.
if (( SLURM_ARRAY_TASK_ID == 0 && SLURM_PROCID == 0 )); then
    mkdir -p "$SWEEP_DIR"
    {
        printf 'name\trs\tgamma\tc\tmov\tvo\n'
        for rs in "${rs_values[@]}"; do
          for g in "${gamma_values[@]}"; do
            for c in "${c_values[@]}"; do
              for m in "${mov_values[@]}"; do
                for v in "${vo_values[@]}"; do
                  printf 'rs%s_g%s_c%s_mov%s_vo%s\t%s\t%s\t%s\t%s\t%s\n' \
                      "$rs" "$g" "$c" "$m" "$v" "$rs" "$g" "$c" "$m" "$v"
                done
              done
            done
          done
        done
    } > "$SWEEP_DIR/configs.tsv"
fi

# The VO geometry varies fastest, so the two arms of the A/B land on adjacent
# task ids and therefore in the same array job.
vo_idx=$((    task_id % num_vo ))
scene_idx=$(( (task_id / num_vo) % num_scenes ))
mov_idx=$((   (task_id / (num_vo * num_scenes)) % num_mov ))
c_idx=$((     (task_id / (num_vo * num_scenes * num_mov)) % num_c ))
gamma_idx=$(( (task_id / (num_vo * num_scenes * num_mov * num_c)) % num_gamma ))
rs_idx=$((     task_id / (num_vo * num_scenes * num_mov * num_c * num_gamma) ))

RS="${rs_values[$rs_idx]}"
GAMMA="${gamma_values[$gamma_idx]}"
C="${c_values[$c_idx]}"
SCENE="${scenes[$scene_idx]}"
MOV="${mov_values[$mov_idx]}"
VO="${vo_values[$vo_idx]}"
# Every swept axis must be in the cell name. Two cells differing only in an
# axis missing from the name would share a directory, and the "already done"
# check below would then skip the second one entirely.
NAME="rs${RS}_g${GAMMA}_c${C}_mov${MOV}_vo${VO}"

# --------------------------------------------------
# 1.5 ROS
# --------------------------------------------------
set +eu
# shellcheck disable=SC1091
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -e

python3 -c 'import rclpy' 2>/dev/null || {
    echo "rclpy still not importable after sourcing ${ROS_SETUP:-/opt/ros/foxy/setup.bash}" >&2
    exit 1
}

# --------------------------------------------------
# 2. Isolation between the tasks sharing this node
# --------------------------------------------------
export ROS_DOMAIN_ID=$(( task_id % 101 ))
export ROS_LOCALHOST_ONLY=1

export NUMBA_CACHE_DIR="/tmp/numba-${task_id}"
export MPLCONFIGDIR="/tmp/mpl-${task_id}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

export HOME="/tmp/home-${task_id}"
export ROS_HOME="$HOME/.ros"
export ROS_LOG_DIR="$ROS_HOME/log"
export XDG_CONFIG_HOME="$HOME/.config"
mkdir -p "$ROS_LOG_DIR" "$XDG_CONFIG_HOME"
export MPLBACKEND=Agg

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"

# --------------------------------------------------
# 3. A working directory per cell
# --------------------------------------------------
CELL="$SWEEP_DIR/$NAME"
WORK="$CELL/work-$SCENE"
mkdir -p "$WORK" "$CELL/logs/$SCENE" "$CELL/debug"

ln -sfn "$REPO/env_build" "$CELL/env_build"
ln -sfn "$REPO/mctsVoRos/MCTS_VO" "$WORK/MCTS_VO"
for f in "$REPO"/mctsVoRos/*.py; do ln -sfn "$f" "$WORK/$(basename "$f")"; done
ln -sfn "$CELL/debug" "$WORK/debug"

cat > "$CELL/config.env" <<EOF
RADIUS_SCALE=$RS
GAMMA=$GAMMA
EXPLORATION_C=$C
VO_GEOMETRY=$VO
MAX_OBS_RADIUS=$MAX_OBS_RADIUS
MAX_OBS_VEL=$MOV
EOF

cd "$WORK"

sleep $(( SLURM_PROCID * 2 ))
python3 -c 'import scipy.linalg, sklearn.cluster, numba' 2>/dev/null \
  || python3 -c 'import scipy.linalg, sklearn.cluster, numba'

echo "task $task_id: $NAME | $SCENE | runs 0..$(( NUM_EXP - 1 ))"
echo "  node $(hostname)  proc $SLURM_PROCID  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  work $WORK"

# --------------------------------------------------
# 4. Run every run number for this cell
# --------------------------------------------------
failed=0
for (( exp_num = 0; exp_num < NUM_EXP; exp_num++ )); do
    log="$CELL/logs/$SCENE/${ALGORITHM}_${exp_num}.log"
    csv="$CELL/debug/$SCENE/data_${ALGORITHM}_${exp_num}.csv"

    if [[ -f "$csv" ]]; then
        echo "  run $exp_num already done"
        continue
    fi

    : > "$log"
    for attempt in 1 2; do
        echo "=== attempt $attempt ===" >> "$log"

        set +e
        python3 loopHandler_copy.py \
            --algorithm "$ALGORITHM" \
            --trajectories "$SCENE" \
            --exp_num "$exp_num" \
            --env-render headless \
            --no-plots \
            --radius-scale "$RS" \
            --gamma-per-second "$GAMMA" \
            --exploration-c "$C" \
            --vo-geometry "$VO" \
            --max-obs-radius "$MAX_OBS_RADIUS" \
            --max-obs-vel "$MOV" \
            >> "$log" 2>&1
        rc=$?
        set -e

        echo "=== exit status $rc ===" >> "$log"
        [[ -f "$csv" ]] && break
        echo "  run $exp_num attempt $attempt: exit $rc, no data" >&2
    done

    if [[ -f "$csv" ]]; then
        echo "  run $exp_num ok"
    else
        echo "  run $exp_num NO DATA - see $log" >&2
        failed=$(( failed + 1 ))
    fi

    # Unity is launched by loopHandler and killed by it on the way out. If a
    # run died before that, the player sits there holding a core. We kill it
    # by matching on the working directory, which is unique per cell+scene.
    for p in $(pgrep -f 'env_build/.*x86_64' 2>/dev/null); do
        if [[ "$(readlink -f /proc/$p/cwd 2>/dev/null)" == "$(readlink -f "$WORK")" ]]; then
            kill "$p" 2>/dev/null || true
        fi
    done
done

echo "task $task_id: $NAME | $SCENE done, $failed of $NUM_EXP produced no data"
[[ $failed -lt $NUM_EXP ]]
