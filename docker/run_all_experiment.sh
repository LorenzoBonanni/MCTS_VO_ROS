#!/usr/bin/env bash
# Full campaign: algorithms x scenes x seeds (fixed parameters).
# One Slurm array task = one specific (algorithm, scene, seed).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-campaign
#SBATCH --time=08:00:00
#SBATCH --array=0-22               # 3 algos * 2 scenes * 30 seeds = 180 tasks; 180/8 = 22.5 → 23 array jobs
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/campaign_%A_%a.out

# Fixed parameters (the paper's defaults, adjust if needed)
export ALGORITHMS="MCTS VO-TREE VO-PLANNER MCTS"
export TRAJECTORIES="sinusoidal_complex intention_complex"   # or "sinusoidal intention"
export NUM_SEEDS=30                 # number of seeds per (algo, scene)
export RADIUS_SCALE=1.4
export GAMMA_PER_SECOND=0.65
export EXPLORATION_C=1.0
export MAX_OBS_VEL=0.25
export ROLLOUT_COLLISION="check"

export SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

# The worker script (must exist in the repo)
srun --environment=mctsvo "$MCTSVO_REPO/docker/run_all_worker.sh"
