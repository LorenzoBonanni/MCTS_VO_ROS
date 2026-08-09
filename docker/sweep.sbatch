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
# GAMMA EXTENSION, ROUND 3: below 0.10, with 0.10 as control.
# 2 rs * 4 gamma * 2 c * 2 scenes * 1 mov * 1 vo = 32 cells, 100 seeds each.
#
# Rounds 1-2 swept 0.10 through 0.65. The result: VO-TREE beats VO-PLANNER on
# both complex scenes once the horizon is short enough, with 0% voluntary
# collisions throughout.
#
#   intention_complex   96% at gamma=0.30 rs=1.4 c=5.0 (VO-PLANNER 76%)
#   sinusoidal_complex  84% at gamma=0.10 rs=1.4 c=5.0 (VO-PLANNER 66%)
#
# The mechanism is the frozen-obstacle model (paper 4.2.1): the search holds
# obstacles where they were observed, so it stays usable only while they have
# not moved far. Effective horizon is dt/(1 - gamma^dt), and the condition is
# v_closing * horizon < r_R + r_i = 0.15 + 0.10 = 0.25 m. That inequality does
# not predict an optimum, it predicts a BREAKING POINT, and intention_complex
# lands on it: 88-92% goal for every gamma where the obstacle covers <= 71% of
# the margin, then 6% at gamma=0.43 where it covers 99%, then 0% beyond.
#
#   gamma/s   0.10  0.15  0.20  0.30  0.43  0.55  0.65
#   horizon    4.9   5.8   6.7   8.8  12.4  17.2  23.7  steps
#   moves/margin 39%  46%   54%   71%   99%  138%  190%
#   intention  90%   88%   92%   88%    6%    0%    0%
#   sinusoidal 79%   56%   49%   12%    3%    3%   26%
#
# sinusoidal_complex degrades monotonically rather than falling off a cliff, and
# was still improving at 0.10 - hence this round. Two purposes: find where it
# stops improving (as the horizon approaches one step VO-TREE degenerates into
# VO-PLANNER, so the point where the curve flattens says whether lookahead
# contributes anything at all), and top the 0.10 cells from 50 to 100 seeds so
# that seeds 50-99 are an out-of-sample check on the 84%, which is the maximum
# of 4 cells and therefore biased upward. Analyse those seeds separately.
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
export GAMMA_VALS="0.02 0.04 0.07 0.10"
export C_VALS="2.0 5.0"
# "paper" is Algorithm 4 as written: tangents to r1, ignore beyond r0 + r1,
# trapped below r1. The comparison against "legacy" (tangents to r0 + r1,
# ignore beyond 1.6*(r0 + r1)) is settled and legacy is dropped: over 4 cells
# x 50 seeds, paper gave 0% voluntary collisions in every cell against legacy's
# 2-4%, with goal rate unchanged. Add "legacy" back to re-run that comparison.
export VO_GEOMETRIES="paper"
# 100: the 0.10 cells already hold seeds 0-49 and the worker skips runs whose
# CSV exists, so this adds seeds 50-99 to them and runs all 100 for the new
# gamma values. Those extra 50 are the out-of-sample check on the 84% cell.
export NUM_EXP=100                # independent seeds per combination
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
