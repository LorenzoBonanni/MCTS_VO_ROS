#!/usr/bin/env bash
# Hyperparameter sweep: rs x gamma x c x scene (fixed algorithm).
# Each cell runs NUM_EXP different seeds (exp_num 0..NUM_EXP-1).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-sweep
#SBATCH --time=06:00:00
# 4 rs * 4 gamma * 4 c * 2 scenes * 2 VO geometries = 256 cells.
# sweep_worker.sh computes task_id = SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID, so
# with --ntasks-per-node=8 the array index runs 0..(256/8 - 1) and the task ids
# tile 0..255 exactly. Do NOT add a step (--array=0-255:16): the multiply by 8
# means a stepped index lands beyond the grid and the worker exits with
# "nothing to do", running a handful of cells instead of all of them. To limit
# how many run at once use %N, which throttles without skipping.
#SBATCH --array=0-31
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/sweep_%A_%a.out

export ALGORITHM="VO-TREE"        # Change to MCTS or VO-PLANNER as needed
export TRAJECTORIES="sinusoidal intention"
export RS_VALS="1.4 1.8 2.2 2.6"
export GAMMA_VALS="0.65 0.81 0.90 0.95"
export C_VALS="0.5 1.0 2.0 5.0"
# The A/B: "paper" is Algorithm 4 as written (tangents to r1, ignore beyond
# r0 + r1, trapped below r1); "legacy" is the geometry the 180-run campaign
# used (tangents to r0 + r1, ignore beyond 1.6*(r0 + r1)). Both geometries see
# the same scenes and the same seeds, so the difference is attributable.
# Set to just "paper" to halve the grid once the comparison is settled.
export VO_GEOMETRIES="paper legacy"
export NUM_EXP=100                 # independent seeds per combination
export MAX_OBS_VEL=0.1
# Largest RANSAC-fitted obstacle radius kept, applied to the measured radius
# before --radius-scale inflates it. Filtering after the scaling, as the code
# used to, made the effective cutoff MAX_OBS_RADIUS/rs, which coupled this to
# the rs axis and made that axis uninterpretable.
export MAX_OBS_RADIUS=0.5
export ROLLOUT_COLLISION="check"

export SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

srun --environment=mctsvo "$MCTSVO_REPO/docker/sweep_worker.sh"
