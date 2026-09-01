#!/usr/bin/env bash
# Hyperparameter sweep, one cell per (rs, gamma, c, scene, mov, vo).
# Each cell runs NUM_EXP seeds.
#
# Current grid: a fairly complete gamma x rs x mov x c sweep on INT_COMPLEX
# alone, post-redesign (the domain is being reworked so the direct start-goal
# line is genuinely blocked - see git log). Nothing here is tuned for that
# redesign yet, this sweep is what tunes it. MOV's floor (0.20) is the OLD
# scene's true obstacle speed; re-check it against the redesigned scene before
# trusting the floor cell - if the new true max differs, the floor rows here
# are silently wrong the same way the pre-585611d ones were.
#
# c (--exploration-c) is in the grid because the shared production value,
# 5.0, was found to be badly oversized: the UCB exploration bonus at c=5.0
# swamps the root Q-value spread by 9-71x across domains (near-uniform action
# selection instead of value-driven), confirmed as a real-outcome effect on
# this exact domain - c=5.0 gave 0% goal-reach, c=0.3 gave 40%, same seeds,
# same scene. Running gamma/rs/mov under a broken c risks measuring the wrong
# regime entirely, not just a worse one, so c is swept jointly rather than
# fixed ahead of time. The 5.0 row stays in as a direct in-sweep reference.
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus-per-task=0
#SBATCH --partition=normal
#SBATCH --job-name=mctsvo-sweep
#SBATCH --time=06:00:00
# 4 tasks/node * 16 cpus/task = 64 cpus/node, same node total as the previous
# 8*8 - half as many concurrent tasks, each with twice the compute, so each
# task's simulation count under the fixed real-time planning budget goes up
# instead of down. sweep_worker.sh:21 hardcodes the "4" in its task_id
# formula to match --ntasks-per-node; change both together or task_id
# silently skips/duplicates grid cells.
#
# Size to the grid: --array=0-(cells/4 - 1). 720 cells here (8 gamma x 6 rs x
# 3 mov x 5 c x 1 scene x 1 vo) -> ceil(720/4)-1 = 179. No step (%N to
# throttle instead): task_id = ARRAY_TASK_ID*4 + PROCID, so a stepped index
# lands past the grid and the worker exits "nothing to do".
#SBATCH --array=0-179
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/sweep_%A_%a.out

# Every axis is overridable. Hardcoding any of them makes an override on the
# sbatch line silently do nothing, which is indistinguishable from a sweep that
# ran and found no effect - RS_VALS and C_VALS were both in that state.
# Remember to resize --array with the grid.
export ALGORITHM="${ALGORITHM:-VO-TREE}"
export TRAJECTORIES="${TRAJECTORIES:-intention_complex}"
export RS_VALS="${RS_VALS:-1.0 1.2 1.4 1.6 1.8 2.0}"
export GAMMA_VALS="${GAMMA_VALS:-0.03 0.06 0.12 0.22 0.35 0.50 0.70 0.90}"
export C_VALS="${C_VALS:-0.05 0.1 0.3 1.0 5.0}"
export VO_GEOMETRIES="${VO_GEOMETRIES:-paper}"   # "legacy" A/B settled, see git log
export NUM_EXP="${NUM_EXP:-30}"
# Must be >= the fastest obstacle in the scene or the VO guarantee fails.
# 0.20 was INT_COMPLEX's true max BEFORE the current redesign (move_1/move_2
# maxSpeed, confirmed by reading the scene directly - see git log). Re-verify
# against the redesigned scene: if the true max moved, the 0.20 row is
# invalid, not just suboptimal - it would silently violate the VO guarantee
# rather than merely score worse.
export MAX_OBS_VEL_VALS="${MAX_OBS_VEL_VALS:-0.20 0.25 0.30}"
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
