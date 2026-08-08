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
# GAMMA EXTENSION: is the discount horizon too long for the frozen-obstacle
# model? 2 rs * 4 gamma * 2 c * 2 scenes * 1 mov * 1 vo = 32 cells.
#
# The search holds obstacles at their observed position (paper 4.2.1), so the
# model is only usable while obstacles have not moved far. gamma sets how far
# ahead the planner relies on it: the effective horizon is dt/(1 - gamma^dt),
# i.e. 2.4 s at 0.65, 4.8 s at 0.81, 9.5 s at 0.90, 19.5 s at 0.95 - and the
# 9600-run sweep found goal rate falling monotonically along exactly that axis,
# 21.2% -> 0.1%. Robot r=0.15 and obstacle r=0.10 collide at 0.25 m, so an
# approaching obstacle consumes the whole margin in 0.25/v_obs seconds: 2.5 s at
# the easy scenes' 0.1 m/s, but only 1.25 s at the complex scenes' 0.2 m/s.
# Solving dt/(1 - gamma^dt) = 1.25 gives gamma ~ 0.43, which the old grid could
# never reach - its floor was 0.65. Hence 0.30/0.43/0.55 with 0.65 as control.
#
# Prediction: a peak near 0.43 on the complex scenes. If instead goal rate keeps
# rising all the way to 0.30, the horizon account is wrong and the tree is
# simply better off not looking ahead at all.
#
# rs and c both take the two per-scene tuned values, since the sweep applies one
# setting to both scenes and they disagree (sin: rs=1.4 c=5.0, int: rs=1.6 c=2.0).
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
# loopHandler_copy.py. These are the _complex variants, which is what every
# sweep so far has actually run: sweep_worker.sh used to hard-code them and
# ignore this export, which read "sinusoidal intention" - the easy scenes.
export TRAJECTORIES="sinusoidal_complex intention_complex"
export RS_VALS="1.4 1.6"
export GAMMA_VALS="0.30 0.43 0.55 0.65"
export C_VALS="2.0 5.0"
# "paper" is Algorithm 4 as written: tangents to r1, ignore beyond r0 + r1,
# trapped below r1. The comparison against "legacy" (tangents to r0 + r1,
# ignore beyond 1.6*(r0 + r1)) is settled and legacy is dropped: over 4 cells
# x 50 seeds, paper gave 0% voluntary collisions in every cell against legacy's
# 2-4%, with goal rate unchanged. Add "legacy" back to re-run that comparison.
export VO_GEOMETRIES="paper"
export NUM_EXP=25                 # independent seeds per combination
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
