#!/usr/bin/env bash
#
# One cell of the sweep: one (rs, gamma, c, scene), every run number for it.
# Launched by sweep.sbatch through srun, one per task on the node.
#
# Set SLURM_ARRAY_TASK_ID and SLURM_PROCID by hand to run a single cell:
#   SLURM_ARRAY_TASK_ID=0 SLURM_PROCID=0 SWEEP_DIR=/scratch/... ./sweep_worker.sh
set -e

# --------------------------------------------------
# 0. Task id
# --------------------------------------------------
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
# 1. The grid. ONE CELL PER TASK: the run number is NOT part of the index, the
#    loop at the bottom covers it.
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
# 2. Isolation between the tasks sharing this node
# --------------------------------------------------
# THIS IS THE ONE THAT WILL BITE. Under docker each run had its own network
# namespace, so Unity and the planner could only ever find their own partner.
# The container engine shares the host network, so eight tasks on a node are on
# one DDS bus: every planner would see every Unity's /scan and /odom. A domain
# id per task puts them on separate ports, and localhost-only keeps discovery
# off the fabric so that tasks on OTHER nodes cannot join either.
export ROS_DOMAIN_ID=$(( SLURM_PROCID % 100 ))
export ROS_LOCALHOST_ONLY=1

# Numba compiles on first use and caches to disk. Eight tasks writing one cache
# directory on a parallel filesystem is a race at best. Node-local, per task.
export NUMBA_CACHE_DIR="/tmp/numba-$SLURM_PROCID"
export MPLCONFIGDIR="/tmp/mpl-$SLURM_PROCID"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

export MPLBACKEND=Agg

# THE OTHER ONE THAT WILL BITE, on a 64-core node. estimate_obstacles builds
# HDBSCAN with n_jobs=-1, so every task tries to use the whole machine, and the
# BLAS underneath numpy does the same. Sixteen tasks each spawning sixty-four
# threads is 1024 threads fighting over 64 cores, and every run gets slower than
# it would have alone. The planner itself is single-threaded, so there is
# nothing to gain by leaving these open.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# joblib, which is what n_jobs=-1 goes through, reads this before falling back
# to the core count.
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"

# --------------------------------------------------
# 3. A working directory per cell
# --------------------------------------------------
# loopHandler_copy.py resolves everything relative to the working directory:
# it writes to 'debug/<scene>/' and launches '../env_build/...'. On the laptop
# each run got its own debug/ by bind-mounting over the repository's; there are
# no per-task bind mounts here, so instead give each cell a directory that
# LOOKS like mctsVoRos/ and is made of symlinks. Costs nothing and keeps the
# results laid out one directory per parameter set, as on the laptop.
CELL="$SWEEP_DIR/$NAME"
# Per SCENE as well as per parameter set: the name deliberately does not carry
# the scene - summarize_sweep.sh wants one directory per parameter set, with
# the scenes as subdirectories of its debug/ - but the two scene tasks of the
# same set run at the same time, so they cannot share a working directory.
WORK="$CELL/work-$SCENE"
mkdir -p "$WORK" "$CELL/logs/$SCENE" "$CELL/debug"

ln -sfn "$REPO/env_build" "$CELL/env_build"
ln -sfn "$REPO/mctsVoRos/MCTS_VO" "$WORK/MCTS_VO"
for f in "$REPO"/mctsVoRos/*.py; do ln -sfn "$f" "$WORK/$(basename "$f")"; done

# debug/ is the parameter set's, shared by both scenes and not the working
# directory's own: loopHandler writes to debug/<scene>/, so they never touch
# the same file, and $CELL/debug is then exactly what summarize_sweep.sh reads.
ln -sfn "$CELL/debug" "$WORK/debug"

cat > "$CELL/config.env" <<EOF
RADIUS_SCALE=$RS
GAMMA=$GAMMA
EXPLORATION_C=$C
EOF

cd "$WORK"

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

    # Resumable: a cell that ran out of wall clock can be resubmitted and picks
    # up where it stopped.
    if [[ -f "$csv" ]]; then
        echo "  run $exp_num already done"
        continue
    fi

    python3 loopHandler_copy.py \
        --algorithm "$ALGORITHM" \
        --trajectories "$SCENE" \
        --exp_num "$exp_num" \
        --env-render headless \
        --no-plots \
        --radius-scale "$RS" \
        --gamma-per-second "$GAMMA" \
        --exploration-c "$C" \
        > "$log" 2>&1 || true

    # loopHandler finishes by raising and catching an exception, so a normal end
    # and a crash look alike from outside. The CSV is the real signal.
    if [[ -f "$csv" ]]; then
        echo "  run $exp_num ok"
    else
        echo "  run $exp_num NO DATA - see $log" >&2
        failed=$(( failed + 1 ))
    fi

    # Unity is launched by loopHandler and killed by it on the way out. If a run
    # died before that, the player sits there holding a core for the rest of the
    # allocation.
    #
    # Matching on the command line is NOT usable here: loopHandler launches the
    # build by a relative path, '../env_build/...', which is identical for every
    # task on the node - a pkill on it would take out fifteen other cells'
    # environments. It also puts the player in its own process group, so it is
    # not reachable as a child either. What IS unique is the working directory
    # it inherited, so match on that.
    for p in $(pgrep -f 'env_build/.*x86_64' 2>/dev/null); do
        if [[ "$(readlink -f /proc/$p/cwd 2>/dev/null)" == "$(readlink -f "$WORK")" ]]; then
            kill "$p" 2>/dev/null || true
        fi
    done
done

echo "task $task_id: $NAME | $SCENE done, $failed of $NUM_EXP produced no data"
[[ $failed -lt $NUM_EXP ]]
