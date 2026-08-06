#!/usr/bin/env bash
set -u

# ------------------------------------------------------------------
# Read grid arrays from environment (set by the sbatch script)
# ------------------------------------------------------------------
IFS=' ' read -ra ALGO_ARR   <<< "${ALGORITHMS:-MCTS VO-TREE VO-PLANNER}"
IFS=' ' read -ra TRAJ_ARR   <<< "${TRAJECTORIES:-sinusoidal intention}"
IFS=' ' read -ra RS_ARR     <<< "${RS_VALS:-1.2 1.5 1.8 2.1}"
IFS=' ' read -ra GAMMA_ARR  <<< "${GAMMA_VALS:-0.65 0.75 0.81 0.9}"
IFS=' ' read -ra C_ARR      <<< "${C_VALS:-0.5 1.0 2.0 5.0}"
N_SEEDS=${NUM_EXP:-20}

N_ALGO=${#ALGO_ARR[@]}
N_TRAJ=${#TRAJ_ARR[@]}
N_RS=${#RS_ARR[@]}
N_GAMMA=${#GAMMA_ARR[@]}
N_C=${#C_ARR[@]}

# ------------------------------------------------------------------
# Unravel the Slurm array index into the five coordinates.
# Order: seed fastest, then c, gamma, rs, trajectory, algorithm slowest.
# (Order matches how the total index is built in the sbatch script.)
# ------------------------------------------------------------------
IDX=$SLURM_ARRAY_TASK_ID

traj_idx=$(( IDX % N_TRAJ ));  IDX=$(( IDX / N_TRAJ ))
rs_idx=$(( IDX % N_RS ));      IDX=$(( IDX / N_RS ))
gamma_idx=$(( IDX % N_GAMMA )); IDX=$(( IDX / N_GAMMA ))
c_idx=$(( IDX % N_C ));        IDX=$(( IDX / N_C ))
algo_idx=$(( IDX % N_ALGO ))

ALGO="${ALGO_ARR[$algo_idx]}"
TRAJ="${TRAJ_ARR[$traj_idx]}"
RS="${RS_ARR[$rs_idx]}"
GAMMA="${GAMMA_ARR[$gamma_idx]}"
C="${C_ARR[$c_idx]}"

# ------------------------------------------------------------------
# Output directory and log file prefix
# ------------------------------------------------------------------
OUT_DIR="${SWEEP_DIR}/${TRAJ}"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ------------------------------------------------------------------
# Loop over seeds (reproducible experiment numbers)
# ------------------------------------------------------------------
for (( seed=0; seed < N_SEEDS; seed++ )); do
    RESULT_CSV="${OUT_DIR}/data_${ALGO}_${seed}.csv"

    # Skip if already completed (resumable)
    if [[ -f "${RESULT_CSV}" ]]; then
        echo "[${ALGO}] [${TRAJ}] [rs=${RS} gamma=${GAMMA} c=${C}] seed=${seed}  SKIP (already done)"
        continue
    fi

    LOG_FILE="${LOG_DIR}/${ALGO}_${TRAJ}_seed${seed}_rs${RS}_gamma${GAMMA}_c${C}.log"
    echo "[${ALGO}] [${TRAJ}] [rs=${RS} gamma=${GAMMA} c=${C}] seed=${seed}  RUNNING"

    # Move to the package directory where loopHandler_copy.py lives
    cd "${MCTSVO_REPO}/mctsVoRos" || exit 1

    # Launch one experiment
    python3 loopHandler_copy.py \
        --exp_num "${seed}" \
        --algorithm "${ALGO}" \
        --trajectories "${TRAJ}" \
        --radius-scale "${RS}" \
        --gamma-per-second "${GAMMA}" \
        --exploration-c "${C}" \
        --max-obs-vel "${MAX_OBS_VEL:-0.25}" \
        --rollout-collision "${ROLLOUT_COLLISION:-check}" \
        --no-plots \
        > "${LOG_FILE}" 2>&1

    status=$?
    if [[ $status -ne 0 ]]; then
        echo "    WARNING: exited with status ${status}, see ${LOG_FILE}"
    fi

    # Kill any leftover Unity process (your fixed cleanup)
    pkill -9 -f "SIN_EASY.x86_64" 2>/dev/null || true
    pkill -9 -f "INT_EASY.x86_64" 2>/dev/null || true
    sleep 2
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished (${ALGO} ${TRAJ} rs=${RS} gamma=${GAMMA} c=${C})"