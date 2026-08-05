#!/usr/bin/env bash
# Parameter sweep, rs x gamma x c x scene, on Slurm.
#
# ONE SRUN TASK = ONE CELL of the grid = one (rs, gamma, c, scene), and it runs
# all NUM_EXP run numbers for that cell in sequence. That is the granularity to
# pick: a single 90 s run per task would make 2560 tasks and the scheduler would
# spend longer placing them than the runs take.
#
#   4 rs x 4 gamma x 4 c x 2 scenes = 128 cells
#   16 cells per node, so --array=0-127:16 = 8 array jobs
#   each cell is NUM_EXP runs of ~90 s, so ~30 min per task at NUM_EXP=20
#
# Cells are independent, so the array can be as wide as the queue allows.
#
# 16 x 4 on a 64-core EPYC 7713. A run is one single-threaded planner plus a
# headless Unity, so four cores each is already slack - the worker pins the
# thread pools to stop HDBSCAN and BLAS claiming the whole node per task.
# Memory is not the constraint: about 2 GB a run against 504 GB on the node,
# so --mem-per-cpu is set well above what is needed and ignored thereafter.
# 8 x 8 works too and leaves more headroom per run; it just doubles the array.
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-sweep
#SBATCH --time=02:00:00
#SBATCH --array=0-127:16
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/mctsvo_%A_%a.out

# PATHS HERE ARE THE CONTAINER'S, not the host's. This script body runs
# outside the container - only the srun step below is inside it - so it must
# not touch these paths itself, just hand them on. /root/sweeps is where the
# EDF mounts ${OUTPUT_SCRATCH}/mctsvo/sweeps; the worker creates what it needs
# once it is inside.
export SWEEP_DIR="${SWEEP_DIR:-/root/sweeps/$SLURM_ARRAY_JOB_ID}"

# The checkout, as the EDF mounts it.
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

export NUM_EXP="${NUM_EXP:-20}"

# --environment names an EDF in ~/.edf/. There is no docker daemon on a compute
# node, so the image has to come through the container engine.
srun --environment=mctsvo "$MCTSVO_REPO/docker/sweep_worker.sh"
