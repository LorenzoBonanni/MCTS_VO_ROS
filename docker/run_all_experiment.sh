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
# 3 algos * 4 scenes * 50 seeds = 600 tasks. run_all_worker.sh computes
# task_id = SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID, so with --ntasks-per-node=8
# the array index runs 0..(600/8 - 1) = 0..74 and the task ids tile 0..599.
#
# RESIZE THIS whenever ALGORITHMS, TRAJECTORIES or NUM_SEEDS changes: an array
# that is too short does not fail, it runs the first 8*(last+1) tasks and stops,
# leaving a campaign that looks finished and is missing whole cells. It was left
# at 0-44 when NUM_SEEDS went 30 -> 50, which would have silently dropped 240
# runs. Too long is harmless - the worker exits "nothing to do".
#SBATCH --array=0-74
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/campaign_%A_%a.out

# MCTS was listed twice here. Since the output file is data_<ALGO>_<seed>.csv,
# the duplicate could never produce anything - the second pass found the CSV
# already present and skipped - but it inflated the task count by 30 per scene.
export ALGORITHMS="${ALGORITHMS:-MCTS VO-TREE VO-PLANNER}"
export TRAJECTORIES="${TRAJECTORIES:-sinusoidal_complex intention_complex sinusoidal intention}"
export NUM_SEEDS="${NUM_SEEDS:-50}"  # seeds per (algo, scene)
# ALGORITHMS/TRAJECTORIES/NUM_SEEDS honour the submitting environment, so a
# single cell can be re-run without editing this file. Size --array to match:
# ceil(N_ALGO * N_TRAJ * NUM_SEEDS / 8) - 1.

# RADIUS_SCALE / GAMMA_PER_SECOND / EXPLORATION_C / MAX_OBS_VEL are deliberately
# NOT exported. run_all_worker.sh selects them per scene from the 9600-run
# sweep - sinusoidal* and intention* want different values - and any export
# here would override that for every scene at once. Export one only to force a
# single value across the whole campaign, e.g. for an ablation.
export MAX_OBS_RADIUS=0.5
export VO_GEOMETRY=paper
export ROLLOUT_COLLISION="check"

export SWEEP_DIR="${SWEEP_DIR:-/root/sweep}"
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

# The worker script (must exist in the repo)
srun --environment=mctsvo "$MCTSVO_REPO/docker/run_all_worker.sh"
