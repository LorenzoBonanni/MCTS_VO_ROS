#!/usr/bin/env bash
# Full campaign: algorithms x trajectories x rs x gamma x c x seeds.
# Each Slurm task is one parameter combination and runs NUM_EXP seeds in a loop.
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-full
#SBATCH --time=08:00:00
#SBATCH --array=0-383:16          # 384 cells total, 16 at a time
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/full_%A_%a.out

# =============================================================================
# Grid dimensions – change these to run a different sweep
# =============================================================================
export ALGORITHMS="MCTS VO-TREE VO-PLANNER"
export TRAJECTORIES="sinusoidal intention"
export RS_VALS="1.2 1.5 1.8 2.1"
export GAMMA_VALS="0.65 0.75 0.81 0.9"
export C_VALS="0.5 1.0 2.0 5.0"
export NUM_EXP=30                  # seeds per configuration

# Fixed parameters for all runs
export MAX_OBS_VEL=0.25
export ROLLOUT_COLLISION="check"  # or "none"

# =============================================================================
# Output directory (on shared scratch)
# =============================================================================
export SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"

# Path to the repository (as mounted inside the container)
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

# The container environment (EDF name)
srun --environment=mctsvo "$MCTSVO_REPO/docker/run_all_worker.sh"