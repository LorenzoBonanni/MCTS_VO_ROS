#!/usr/bin/env python3
"""
Generate the ground-truth position GIFs for every run under debug/ that
doesn't have them yet, then sort every animation file for that run -
trajectory_<suffix>.gif, rollout_<suffix>.mp4 (MCTS/VO-TREE only),
positions_<suffix>.gif, positions_estimates_<suffix>.gif - into
debug/sorted/<SCENE>/<ALGORITHM>/<OUTCOME>/, moving them out of the flat
debug/<scene>/animations/ directory.

Reuses summarize_debug.discover()/outcome_of() for run discovery and outcome
classification (identical precedence and CSV schema tolerance as every other
report in this repo), and position_log.animate_run_positions[_with_estimates]
for the GIFs themselves - this script adds no new logic for either.

Resumable: a run is skipped if its target directory already has a
positions_<suffix>.gif, so re-running after an interruption only processes
what's left. Run from mctsVoRos/ (matching every other script here):

    docker/run.sh python3 organize_debug_animations.py

Runs are independent (each writes only its own files, keyed by its own
suffix), so they're rendered in a process pool - one process per CPU core.
Multiprocessing, not threading: matplotlib/pyplot keeps global figure state
that isn't thread-safe, but is fine across separate processes. MPLBACKEND=Agg
is set at the image level (docker/Dockerfile), so no display is needed in any
worker.
"""
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

from position_log import animate_run_positions, animate_run_positions_with_estimates
from summarize_debug import discover, outcome_of

DEBUG_ROOT = "debug"
SORTED_ROOT = os.path.join(DEBUG_ROOT, "sorted")

OUTCOME_DIR = {
    "success": "SUCCESS",
    "collision": "VOL_COLLISION",
    "obsCollision": "OBS_COLLISION",
    "timeout": "TIMEOUT",
}

# The four animation file types a run can have. rollout_*.mp4 only exists for
# MCTS/VO-TREE (VO-PLANNER has no tree to animate); the rest are always
# attempted and simply skipped if absent.
ANIMATION_PATTERNS = [
    "trajectory_{suffix}.gif",
    "rollout_{suffix}.mp4",
    "positions_{suffix}.gif",
    "positions_estimates_{suffix}.gif",
]


def target_dir(scene, algorithm, outcome):
    return os.path.join(SORTED_ROOT, scene.upper(), algorithm, OUTCOME_DIR[outcome])


def move_animations(scene, algorithm, suffix, dest):
    anim_dir = os.path.join(DEBUG_ROOT, scene, "animations")
    moved = 0
    for pattern in ANIMATION_PATTERNS:
        src = os.path.join(anim_dir, pattern.format(suffix=suffix))
        if os.path.exists(src):
            shutil.move(src, os.path.join(dest, os.path.basename(src)))
            moved += 1
    return moved


def _silence_tqdm():
    # Each animate_run_positions[_with_estimates] call drives its own tqdm
    # progress bar; with a dozen-plus worker processes writing to the same
    # terminal at once those interleave into unreadable noise. Only the
    # per-run done/skip/FAIL lines from process_run are worth seeing when
    # running in parallel.
    import position_log
    position_log.tqdm = lambda iterable, **kwargs: iterable


def process_run(run):
    scene = run.trajectories
    algorithm = run.algorithm
    exp_num = run.exp_num
    suffix = f"{algorithm}_{exp_num}{run.suffix}"
    outcome = outcome_of(run)
    dest = target_dir(scene, algorithm, outcome)

    positions_gif = os.path.join(dest, f"positions_{suffix}.gif")
    if os.path.exists(positions_gif):
        print(f"skip  {scene}/{algorithm}/{exp_num} ({outcome}) - already done")
        return

    os.makedirs(dest, exist_ok=True)

    try:
        animate_run_positions(scene, algorithm, exp_num, run.suffix, DEBUG_ROOT)
        animate_run_positions_with_estimates(scene, algorithm, exp_num, run.suffix, DEBUG_ROOT)
    except FileNotFoundError as exc:
        print(f"SKIP  {scene}/{algorithm}/{exp_num} ({outcome}) - {exc}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"FAIL  {scene}/{algorithm}/{exp_num} ({outcome}) - {exc}", file=sys.stderr)
        return

    moved = move_animations(scene, algorithm, suffix, dest)
    print(f"done  {scene}/{algorithm}/{exp_num} ({outcome}) - {moved} files -> {dest}")


def main():
    runs = discover(DEBUG_ROOT)
    print(f"{len(runs)} runs found")
    workers = max((os.cpu_count() or 4) // 2, 1)
    print(f"rendering with {workers} worker processes")
    with ProcessPoolExecutor(max_workers=workers, initializer=_silence_tqdm) as pool:
        # list() drains the iterator so we actually wait for every run; each
        # worker prints its own done/skip/FAIL line as it finishes.
        list(pool.map(process_run, runs))


if __name__ == "__main__":
    main()
