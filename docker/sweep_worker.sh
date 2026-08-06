#!/usr/bin/env bash
#
# One cell of the sweep: one (rs, gamma, c, scene), every run number for it.
set -e

if [[ -z "${SLURM_ARRAY_TASK_ID}" || -z "${SLURM_PROCID}" ]]; then
    echo "expects SLURM_ARRAY_TASK_ID and SLURM_PROCID to be set" >&2
    exit 1
fi
task_id=$(( SLURM_ARRAY_TASK_ID + SLURM_PROCID ))

: "${SWEEP_DIR:?export SWEEP_DIR}"
REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"
NUM_EXP="${NUM_EXP:-20}"
ALGORITHM="${ALGORITHM:-VO-TREE}"

# --------------------------------------------------
# 1. The grid.
# --------------------------------------------------
rs_values=(1.4 1.8 2.2 2.6)
gamma_values=(0.65 0.81 0.90 0.95)
c_values=(0.5 1.0 2.0 5.0)
scenes=(sinusoidal intention)

num_rs=${#rs_values[@]}
num_gamma=${#gamma_values[@]}
num_c=${#c_values[@]}
num_scenes=${#scenes[@]}
total_cells=$(( num_rs * num_gamma * num_c * num_scenes ))

if (( task_id >= total_cells )); then
    echo "task_id $task_id beyond the grid (0..$(( total_cells - 1 ))), nothing to do"
    exit 0
fi

if (( SLURM_ARRAY_TASK_ID == 0 && SLURM_PROCID == 0 )); then
    mkdir -p "$SWEEP_DIR"
    {
        printf 'name\trs\tgamma\tc\n'
        for rs in "${rs_values[@]}"; do
          for g in "${gamma_values[@]}"; do
            for c in "${c_values[@]}"; do
                printf 'rs%s_g%s_c%s\t%s\t%s\t%s\n' "$rs" "$g" "$c" "$rs" "$g" "$c"
            done
          done
        done
    } > "$SWEEP_DIR/configs.tsv"
fi

rs_idx=$((   task_id / (num_gamma * num_c * num_scenes) ))
gamma_idx=$(( (task_id / (num_c * num_scenes)) % num_gamma ))
c_idx=$((     (task_id / num_scenes) % num_c ))
scene_idx=$(( task_id % num_scenes ))

RS="${rs_values[$rs_idx]}"
GAMMA="${gamma_values[$gamma_idx]}"
C="${c_values[$c_idx]}"
SCENE="${scenes[$scene_idx]}"
NAME="rs${RS}_g${GAMMA}_c${C}"

# --------------------------------------------------
# 1.5 ROS (unchanged)
# --------------------------------------------------
set +eu
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -e

python3 -c 'import rclpy' 2>/dev/null || {
    echo "rclpy still not importable after sourcing ${ROS_SETUP:-/opt/ros/foxy/setup.bash}" >&2
    exit 1
}

# --------------------------------------------------
# 2. Isolation (unchanged) – the domain id, HOME, Numba, thread pinning, etc.
# --------------------------------------------------
export ROS_DOMAIN_ID=$(( (SLURM_ARRAY_TASK_ID + SLURM_PROCID) % 101 ))
export ROS_LOCALHOST_ONLY=1
export NUMBA_CACHE_DIR="/tmp/numba-$SLURM_PROCID"
export MPLCONFIGDIR="/tmp/mpl-$SLURM_PROCID"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
export HOME="/tmp/home-$SLURM_PROCID"
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
# 3. Working directory per cell (unchanged)
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
EOF

cd "$WORK"

sleep $(( SLURM_PROCID * 2 ))
python3 -c 'import scipy.linalg, sklearn.cluster, numba' 2>/dev/null \
  || python3 -c 'import scipy.linalg, sklearn.cluster, numba'

echo "task $task_id: $NAME | $SCENE | runs 0..$(( NUM_EXP - 1 ))"
echo "  node $(hostname)  proc $SLURM_PROCID  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  work $WORK"

# --------------------------------------------------
# 4. Run every run number – NOW WITH UNIQUE SEED
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

    # Kill Unity using the working-directory trick (unchanged)
    for p in $(pgrep -f 'env_build/.*x86_64' 2>/dev/null); do
        if [[ "$(readlink -f /proc/$p/cwd 2>/dev/null)" == "$(readlink -f "$WORK")" ]]; then
            kill "$p" 2>/dev/null || true
        fi
    done
done

echo "task $task_id: $NAME | $SCENE done, $failed of $NUM_EXP produced no data"
[[ $failed -lt $NUM_EXP ]]