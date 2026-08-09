#!/usr/bin/env bash
# Full campaign: algorithms x scenes x seeds. One array task = one run.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-campaign
#SBATCH --time=00:30:00
# RESIZE with the grid: --array=0-(N_ALGO*N_TRAJ*NUM_SEEDS/8 - 1), here 2400
# tasks -> 0-299. Too short does not fail, it silently runs a prefix and stops.
#SBATCH --array=0-299
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/campaign_%A_%a.out

export ALGORITHMS="${ALGORITHMS:-MCTS VO-TREE VO-PLANNER}"
export TRAJECTORIES="${TRAJECTORIES:-sinusoidal_complex intention_complex sinusoidal intention}"
# 200, so that 0% vs 3% voluntary collisions separates at p<0.05. That
# comparison - VO against plain MCTS at the corrected gamma - is the point.
export NUM_SEEDS="${NUM_SEEDS:-200}"

# RADIUS_SCALE / GAMMA_PER_SECOND / EXPLORATION_C / MAX_OBS_VEL are chosen per
# scene in run_all_worker.sh. Exporting one here overrides it for every scene.
export MAX_OBS_RADIUS=0.5
export VO_GEOMETRY=paper
export RANGE_METRIC="${RANGE_METRIC:-norm}"   # "width" is the F2 fix
export ROLLOUT_COLLISION="check"

export SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

srun --environment=mctsvo "$MCTSVO_REPO/docker/run_all_worker.sh"
