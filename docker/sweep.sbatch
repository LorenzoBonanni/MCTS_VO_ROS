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
# EASY SCENES: the gamma sweep they never got.
# 2 rs * 4 gamma * 2 c * 2 scenes * 1 mov * 1 vo = 32 cells, 50 seeds each.
#
# Rounds 1-3 swept the COMPLEX scenes (obstacles at 0.2 m/s) and settled on
# 0.04/s for sinusoidal* and 0.30/s for intention*. Those values were then
# extrapolated to the easy scenes by the sinusoidal*/intention* glob in
# run_all_worker.sh, on the argument that halving the obstacle speed leaves the
# frozen-obstacle model comfortably valid. True, but it only considered the
# horizon being too LONG. The 2400-run campaign showed the opposite failure:
#
#   intention/VO-TREE     16% goal, 83% obsColl   (was 48% at gamma=0.65)
#   sinusoidal/VO-TREE    91% goal                (was 98% at DEPTH=200)
#
# At 0.1 m/s an obstacle covers only 35% of the 0.25 m margin over a 0.88 s
# horizon, so validity is not the binding constraint on these scenes - the
# planner is simply blind. The rule is the LONGEST horizon that keeps
# v_closing * horizon inside the margin, which at 0.1 m/s allows about 2.5 s,
# i.e. gamma up to ~0.67. Hence 0.30 (the current value, as control) through
# 0.80, which is deliberately past that boundary so the cliff is visible if
# there is one.
#
# Note gamma now sets the rollout budget too, since DEPTH is derived from the
# discount: 115 steps at 0.30 and the 200-step cap at 0.50 and above. So this
# sweep moves depth as well as horizon, which is the point - the easy scenes
# inherited both from scenes with twice the obstacle speed.
#
# rs and c span the values the complex scenes settled on, since the easy scenes
# have no tuning of their own and the sweep applies one setting to both.
# Full grid, to restore:
#   RS_VALS="1.2 1.4 1.6"  GAMMA_VALS="0.65 0.81 0.90 0.95"
#   C_VALS="0.5 1.0 2.0 5.0"  MAX_OBS_VEL_VALS="0.2 0.25"
#   MAX_OBS_RADIUS=0.5  NUM_EXP=50  --array=0-23   (192 cells)
# sweep_worker.sh computes task_id = SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID, so
# with --ntasks-per-node=8 the array index runs 0..(192/8 - 1) and the task ids
# tile 0..191 exactly. Do NOT add a step (--array=0-191:16): the multiply by 8
# means a stepped index lands beyond the grid and the worker exits with
# "nothing to do", running a handful of cells instead of all of them. To limit
# how many run at once use %N, which throttles without skipping.
#SBATCH --array=0-3
#SBATCH --output=/capstor/scratch/cscs/lbonanni/logs/sweep_%A_%a.out

export ALGORITHM="VO-TREE"        # Change to MCTS or VO-PLANNER as needed
# The scene names are directory names under debug/ and keys of ENV_BUILDS in
# loopHandler_copy.py. The EASY variants here: 2 moving obstacles at 0.1 m/s,
# against 4 at up to 0.2 in the _complex ones. They are the only scenes whose
# gamma has never been swept.
export TRAJECTORIES="sinusoidal intention"
export RS_VALS="1.4 1.6"
export GAMMA_VALS="0.30 0.50 0.65 0.80"
export C_VALS="2.0 5.0"
# "paper" is Algorithm 4 as written: tangents to r1, ignore beyond r0 + r1,
# trapped below r1. The comparison against "legacy" (tangents to r0 + r1,
# ignore beyond 1.6*(r0 + r1)) is settled and legacy is dropped: over 4 cells
# x 50 seeds, paper gave 0% voluntary collisions in every cell against legacy's
# 2-4%, with goal rate unchanged. Add "legacy" back to re-run that comparison.
export VO_GEOMETRIES="paper"
# 50 is enough to place the curve; the best cell then wants an out-of-sample
# check before it is quoted, as every previous round's maximum regressed.
export NUM_EXP=50                 # independent seeds per combination
# Swept axis. Must be >= the fastest obstacle in the scene or the VO safety
# guarantee does not hold. The .unity files override the move_*.cs defaults:
# max_speed/maxSpeed are 0.2 on the fastest instances in both complex scenes,
# so 0.2 is the tight bound and 0.25 adds margin. An earlier sweep ran at 0.1 -
# half the true maximum - which undersized every collision cone.
export MAX_OBS_VEL_VALS="0.25"
# Largest RANSAC-fitted obstacle radius kept, applied to the measured radius
# before --radius-scale inflates it. Filtering after the scaling, as the code
# used to, made the effective cutoff MAX_OBS_RADIUS/rs, which coupled this to
# the rs axis and made that axis uninterpretable.
# Back to 0.5 after the perception probe left this at 5.0 (effectively
# disabled). The probe is settled - it refuted MAX_OBS_RADIUS as the cause of
# the detection dropouts, 2.25% voluntary collisions against a 2.17% baseline -
# and 0.5 is the value the campaign and every tuning result use.
export MAX_OBS_RADIUS=0.5
export ROLLOUT_COLLISION="check"

# Hardcoded, NOT "${SWEEP_DIR:-...}". This path is a bind mount defined in
# ~/.edf/mctsvo.toml ("${OUTPUT_SCRATCH}/sweep:/root/sweep") and is the only
# writable path inside the container that survives the job. Honouring an
# inherited SWEEP_DIR once cost a full 192-cell sweep: a host path was exported
# in the submitting shell, sbatch passed it through, and because that path is
# not mounted the container silently created it in its own writable layer and
# threw away all 9600 runs on exit. Archive the contents between sweeps rather
# than pointing this elsewhere.
export SWEEP_DIR=/root/sweep
export MCTSVO_REPO="${MCTSVO_REPO:-/workspace/MCTS_VO_ROS}"

srun --environment=mctsvo "$MCTSVO_REPO/docker/sweep_worker.sh"
