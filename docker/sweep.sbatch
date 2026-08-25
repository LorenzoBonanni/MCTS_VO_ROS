#!/usr/bin/env bash
# Hyperparameter sweep, one cell per (rs, gamma, c, scene, mov, vo).
# Each cell runs NUM_EXP seeds.
#
# Current grid: max_obs_vel on the easy scenes, at each scene's own gamma.
# They want different gammas, which one grid cannot express, so run two arms:
#   TRAJECTORIES=intention  GAMMA_VALS=0.50 sbatch docker/sweep.sbatch
#   TRAJECTORIES=sinusoidal GAMMA_VALS=0.65 sbatch docker/sweep.sbatch
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-sweep
#SBATCH --time=06:00:00
# Size to the grid: --array=0-(cells/8 - 1), here 12 cells. No step (%N to
# throttle instead): task_id = ARRAY_TASK_ID*8 + PROCID, so a stepped index
# lands past the grid and the worker exits "nothing to do".
#SBATCH --array=0-1
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/sweep_%A_%a.out

# Every axis is overridable. Hardcoding any of them makes an override on the
# sbatch line silently do nothing, which is indistinguishable from a sweep that
# ran and found no effect - RS_VALS and C_VALS were both in that state.
# Remember to resize --array with the grid.
export ALGORITHM="${ALGORITHM:-VO-TREE}"
export TRAJECTORIES="${TRAJECTORIES:-intention}"
export RS_VALS="${RS_VALS:-1.4 1.6}"
export GAMMA_VALS="${GAMMA_VALS:-0.50}"
export C_VALS="${C_VALS:-2.0 5.0}"
export VO_GEOMETRIES="${VO_GEOMETRIES:-paper}"   # "legacy" A/B settled, see git log
export NUM_EXP="${NUM_EXP:-50}"
# Must be >= the fastest obstacle in the scene or the VO guarantee fails.
# True maxima: 0.1 m/s on the easy scenes, 0.2 on the _complex ones.
export MAX_OBS_VEL_VALS="${MAX_OBS_VEL_VALS:-0.125 0.15 0.25}"
export MAX_OBS_RADIUS="${MAX_OBS_RADIUS:-0.5}"   # applied before --radius-scale
export ROLLOUT_COLLISION="check"

# Hardcoded, not "${SWEEP_DIR:-...}": this is the only bind mount that survives
# the job (~/.edf/mctsvo.toml). An inherited host path once cost a 9600-run
# sweep - the container created it in its writable layer and discarded it.
export SWEEP_DIR=/root/sweep
# The EDF binds $OUTPUT_SCRATCH/sweep here. pyxis cannot mount a source that
# does not exist: it fails before the container starts, so every task dies in
# ~1s with no log of its own. Deleting the tree to reclaim Lustre inodes is a
# normal thing to do between sweeps, so recreate it rather than rely on it.
mkdir -p "$OUTPUT_SCRATCH/sweep"
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

srun --environment=mctsvo "$MCTSVO_REPO/docker/sweep_worker.sh"
