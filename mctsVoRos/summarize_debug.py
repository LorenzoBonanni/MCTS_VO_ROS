#!/usr/bin/env python3
"""Summarize the experiment artefacts under a debug/ folder.

Reads every data_<ALGO>_<N><suffix>.csv it can find, groups the runs by
configuration, and prints outcome, return, timing, depth and smoothness tables.
Also snapshots a debug folder under a label, so that stepping through a series
of commits does not overwrite the previous step's results.

See SUMMARIZE_README.md for what the numbers mean.

Deliberately untracked: it lives next to the code it reads but is never part of
a commit, so checking out or cherry-picking never touches it.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# The goal is a constant in loopHandler_copy.py and is never written to disk, so
# plotting has to be told where it was. This is the corrected value; runs made
# before the goal-coordinate fix were aimed at [-3.26, -1.61] instead. The plot
# command cross-checks whichever value is used against where the successful runs
# actually stopped, and complains if they disagree.
DEFAULT_GOAL = (-2.783, -0.720)

# data_VO-TREE_12_hz05.csv -> ("VO-TREE", "12", "_hz05")
RUN_RE = re.compile(r"^data_([A-Za-z-]+)_(\d+)(.*)\.csv$")

# `snapshot` archives every run artefact, not just the ones the summary reads,
# because it deletes the originals afterwards - anything left out would be lost
# rather than merely absent. Measured on a full campaign: these come to ~50 MB,
# against 210 MB for the rendered animations, which is why those are separate.
#
# obs_ in particular must be here: the plots draw the detected obstacles from
# it, so an archive without it can produce summaries but not pictures.
SNAPSHOT_PREFIXES = (
    "data_", "step_stats_", "times_", "sim_num_", "depths_",
    "acts_", "actions_executed_", "trj_", "obs_", "obsPred_", "ps_",
)

# Rendered GIFs and MP4s. ~210 MB of a 260 MB campaign folder, and regenerable
# from the pickles, so they are archived only on request - and then not deleted.
ANIMATION_DIR = "animations"

# Robot limits, used only to bound-check the smoothness figures.
MAX_SPEED = 0.22          # m/s
MAX_ANGLE_RATE = 2.84     # rad/s


# ---------------------------------------------------------------- discovery --

class Run:
    """One experiment run: its CSV row plus lazily-loaded per-step pickles."""

    def __init__(self, path, algorithm, exp_num, suffix, row, source):
        self.path = path
        self.dir = os.path.dirname(path)
        self.algorithm = algorithm
        self.exp_num = exp_num
        self.suffix = suffix
        self.row = row
        self.source = source          # snapshot label, or "" for the live folder
        self._cache = {}

    def aux(self, prefix):
        """Load <prefix><ALGO>_<N><suffix>.pkl, or None if absent/unreadable."""
        if prefix in self._cache:
            return self._cache[prefix]
        p = os.path.join(
            self.dir, f"{prefix}{self.algorithm}_{self.exp_num}{self.suffix}.pkl")
        obj = None
        if os.path.exists(p):
            try:
                with open(p, "rb") as fh:
                    obj = pickle.load(fh)
            except Exception as exc:                      # truncated / killed mid-write
                warn(f"unreadable {os.path.basename(p)}: {exc}")
        self._cache[prefix] = obj
        return obj

    def get(self, col, default=np.nan):
        """CSV field, tolerating the older schema that lacks the later columns."""
        if col not in self.row or pd.isna(self.row[col]):
            return default
        return self.row[col]

    @property
    def trajectories(self):
        v = self.get("trajectories", None)
        return v if isinstance(v, str) else os.path.basename(self.dir)

    @property
    def ts(self):
        """Control step. Present as a column only from the C6 commit onward."""
        v = self.get("ts", None)
        return float(v) if v is not None and not pd.isna(v) else None


def warn(msg):
    print(f"  ! {msg}", file=sys.stderr)


def rel(path, base):
    """Path relative to base, for output that is readable rather than absolute."""
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def discover(root, label=""):
    """Every run under root/<trajectories>/, plus root/ itself."""
    runs = []
    if not os.path.isdir(root):
        return runs
    subdirs = [os.path.join(root, d) for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]
    for d in subdirs + [root]:
        for name in sorted(os.listdir(d)):
            m = RUN_RE.match(name)
            if not m:
                continue
            path = os.path.join(d, name)
            try:
                df = pd.read_csv(path)
                if len(df) == 0:
                    warn(f"empty {name}, skipped")
                    continue
                row = df.iloc[0]
            except Exception as exc:                       # killed mid-write
                warn(f"unreadable {name}: {exc}, skipped")
                continue
            runs.append(Run(path, m.group(1), m.group(2), m.group(3), row, label))
    return runs


# ------------------------------------------------------------------ metrics --

def rate(runs, col):
    """Percentage of runs where a boolean CSV column is true."""
    vals = [bool(r.get(col, False)) for r in runs]
    return 100.0 * sum(vals) / len(vals) if vals else np.nan


def mean_std(runs, col):
    v = np.array([r.get(col, np.nan) for r in runs], dtype=float)
    v = v[~np.isnan(v)]
    return (np.mean(v), np.std(v)) if len(v) else (np.nan, np.nan)


def timing(runs, legacy_ts):
    """Per-phase timing, measured where possible and reconstructed otherwise.

    step_stats_*.pkl exists only from the C1 commit onward. Before it, the loop
    stored times[i] = dt - t_sense - 0.005, so the split is still recoverable
    exactly rather than approximately:

        t_plan  = times[i]
        t_sense = dt - 0.005 - times[i]
        t_cycle = 2 * dt            (t_timer was hard-coded to 2*dt before C6)

    Returns (dict of seconds, derived_flag).
    """
    sense, plan, cycle, sims = [], [], [], []
    derived = False

    for r in runs:
        st = r.aux("step_stats_")
        if st:
            sense += [s["t_sense"] for s in st]
            plan += [s["t_plan"] for s in st]
            cycle += [s["t_cycle"] for s in st if not np.isnan(s.get("t_cycle", np.nan))]
            sims += [s["n_sims"] for s in st if not np.isnan(s.get("n_sims", np.nan))]
            continue

        derived = True
        ts = r.ts or legacy_ts
        times = r.aux("times_")
        if times:
            plan += list(times)
            sense += [ts - 0.005 - t for t in times]
            cycle += [2.0 * ts] * len(times)
        sn = r.aux("sim_num_")                 # absent for VO-PLANNER: no tree
        if sn:
            sims += list(sn)

    def med_p99(v):
        a = np.array(v, dtype=float)
        a = a[~np.isnan(a)]
        if not len(a):
            return np.nan, np.nan
        return float(np.median(a)), float(np.percentile(a, 99))

    s_med, s_p99 = med_p99(sense)
    p_med, p_p99 = med_p99(plan)
    c_med, c_p99 = med_p99(cycle)
    steps, _ = mean_std(runs, "nSteps")
    return {
        "t_sense_med": s_med, "t_sense_p99": s_p99,
        "t_plan_med": p_med, "t_plan_p99": p_p99,
        "t_cycle_med": c_med, "t_cycle_p99": c_p99,
        "hz": 1.0 / c_med if c_med and not np.isnan(c_med) and c_med > 0 else np.nan,
        "total_s": c_med * steps if not np.isnan(c_med) and not np.isnan(steps) else np.nan,
        "sims": float(np.median(sims)) if sims else np.nan,
    }, derived


def smoothness(runs, legacy_ts):
    """The paper's two smoothness measures, plus context for reading them.

    Both are defined on the *commanded* signal (ROBOT-D-25-00227, p. 22) and
    both are minimised:

        m_vsm = 1/(T-1) sum ||v_{t+1} - v_t||_2 / t_s     [m/s^2]
        m_hsm = 1/(T-1) sum |w_{t+1} - w_t|    / t_s      [rad/s^2]

    v is the velocity *vector* <v cos a, v sin a>, so a change of heading at
    constant speed contributes - which a scalar |dv| would miss. acts_ holds
    [speed, heading] and actions_executed_ holds [speed, omega] as published to
    cmd_vel.

    Divide by t_s, never by the step index: a step is 0.1 s at ts=0.1 and 0.02 s
    at ts=0.02, and without this the numbers stop being comparable the moment
    the control period changes.
    """
    vsm, hsm, stops, revs, lengths, effs = [], [], [], [], [], []

    for r in runs:
        ts = r.ts or legacy_ts

        acts = r.aux("acts_")
        if acts is not None and len(acts) >= 2:
            a = np.asarray(acts, dtype=float)
            vec = np.column_stack((a[:, 0] * np.cos(a[:, 1]),
                                   a[:, 0] * np.sin(a[:, 1])))
            vsm.append(np.mean(np.linalg.norm(np.diff(vec, axis=0), axis=1)) / ts)
            stops.append(100.0 * np.mean(a[:, 0] == 0.0))
            sign = np.sign(a[:, 0])
            nz = sign[sign != 0]
            revs.append(int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0)

        ex = r.aux("actions_executed_")
        if ex is not None and len(ex) >= 2:
            w = np.asarray(ex, dtype=float)[:, 1]
            hsm.append(np.mean(np.abs(np.diff(w))) / ts)

        trj = r.aux("trj_")
        if trj is not None and len(trj) >= 2:
            xy = np.asarray(trj, dtype=float)[:, :2]
            L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            lengths.append(L)
            straight = float(np.linalg.norm(xy[-1] - xy[0]))
            if L > 1e-9:
                effs.append(min(straight / L, 1.0))

    m = lambda v: float(np.mean(v)) if v else np.nan
    return {"m_vsm": m(vsm), "m_hsm": m(hsm), "stop_pct": m(stops),
            "reversals": m(revs), "path_len": m(lengths), "path_eff": m(effs)}


# ------------------------------------------------------------------- output --

def fmt(v, spec=".2f"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or math.isinf(v))):
        return "-"
    return format(v, spec)


def table(title, headers, rows, note=None):
    if not rows:
        return
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(c[i]) for c in cols) for i in range(len(headers))]
    print(f"\n{title}")
    print("  " + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(c).rjust(w) for c, w in zip(r, widths)))
    if note:
        print(f"  {note}")


def group_key(r):
    return (r.source, r.trajectories, r.algorithm, r.suffix)


def group_label(key):
    source, traj, algo, suffix = key
    name = f"{algo}{suffix}"
    return f"{source}/{traj}/{name}" if source else f"{traj}/{name}"


def summarize(runs, legacy_ts, csv_out=None, show_runs=False, runs_csv_out=None):
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    keys = sorted(groups)

    print(f"\n{len(runs)} runs in {len(keys)} groups")

    out_rows, ret_rows, tim_rows, dep_rows, smo_rows, wide = [], [], [], [], [], []
    any_derived = False
    gammas, ts_values = set(), set()

    for k in keys:
        g = groups[k]
        label = group_label(k)
        n = len(g)

        for r in g:
            gm, tsv = r.get("gammaPerSecond", None), r.get("ts", None)
            gammas.add(round(float(gm), 6) if gm is not None and not pd.isna(gm) else None)
            ts_values.add(round(float(tsv), 6) if tsv is not None and not pd.isna(tsv) else None)

        steps_m, _ = mean_std(g, "nSteps")
        out_rows.append([label, n,
                         fmt(rate(g, "reachGoal"), ".0f"),
                         fmt(rate(g, "collision"), ".0f"),
                         fmt(rate(g, "Obscollision"), ".0f"),
                         fmt(rate(g, "maxSteps"), ".0f"),
                         fmt(steps_m, ".0f")])

        dm, ds = mean_std(g, "discountedReturn")
        um, us = mean_std(g, "undiscountedReturn")
        ret_rows.append([label, n, f"{fmt(dm)} +- {fmt(ds)}", f"{fmt(um)} +- {fmt(us)}"])

        t, derived = timing(g, legacy_ts)
        any_derived |= derived
        tim_rows.append([label + (" *" if derived else ""),
                         fmt(t["t_sense_med"] * 1e3, ".3f") if not np.isnan(t["t_sense_med"]) else "-",
                         fmt(t["t_sense_p99"] * 1e3, ".3f") if not np.isnan(t["t_sense_p99"]) else "-",
                         fmt(t["t_plan_med"] * 1e3, ".1f") if not np.isnan(t["t_plan_med"]) else "-",
                         fmt(t["t_cycle_med"] * 1e3, ".1f") if not np.isnan(t["t_cycle_med"]) else "-",
                         fmt(t["hz"], ".1f"), fmt(t["total_s"], ".0f"), fmt(t["sims"], ".0f")])

        dep_rows.append([label,
                         fmt(mean_std(g, "meanTreeDepth")[0], ".1f"),
                         fmt(mean_std(g, "maxTreeDepth")[0], ".0f"),
                         fmt(mean_std(g, "meanRolloutDepth")[0], ".1f"),
                         fmt(mean_std(g, "maxRolloutDepth")[0], ".0f"),
                         fmt(mean_std(g, "meanTotalDepth")[0], ".1f"),
                         fmt(mean_std(g, "maxTotalDepth")[0], ".0f")])

        s = smoothness(g, legacy_ts)
        smo_rows.append([label, fmt(s["m_vsm"], ".3f"), fmt(s["m_hsm"], ".3f"),
                         fmt(s["stop_pct"], ".1f"), fmt(s["reversals"], ".1f"),
                         fmt(s["path_len"], ".2f"), fmt(s["path_eff"], ".3f")])

        wide.append(dict(
            group=label, source=k[0], trajectories=k[1], algorithm=k[2],
            suffix=k[3], n=n,
            goalPct=rate(g, "reachGoal"), volCollPct=rate(g, "collision"),
            obsCollPct=rate(g, "Obscollision"), timeoutPct=rate(g, "maxSteps"),
            nSteps=steps_m, discReturn=dm, discReturnStd=ds,
            undiscReturn=um, undiscReturnStd=us, timingDerived=derived,
            **{kk: vv for kk, vv in t.items()}, **s))

    table("OUTCOMES", ["group", "n", "goal%", "volColl%", "obsColl%", "timeout%", "steps"],
          out_rows,
          "volColl = robot drove into an obstacle (what VO must prevent);"
          " obsColl = obstacle hit a stopped robot.")

    table("RETURNS", ["group", "n", "discounted", "undiscounted"], ret_rows)

    table("TIMING (ms)", ["group", "senseMed", "senseP99", "planMed", "cycleMed",
                          "Hz", "totalS", "sims/step"], tim_rows,
          "* = reconstructed from times_*.pkl, not measured "
          "(runs predating the step_stats instrumentation)." if any_derived else None)

    table("DEPTH", ["group", "treeMean", "treeMax", "rollMean", "rollMax",
                    "totMean", "totMax"], dep_rows,
          "blank for VO-PLANNER, which has no tree.")

    table("SMOOTHNESS", ["group", "m_vsm", "m_hsm", "stop%", "revers", "pathLen", "pathEff"],
          smo_rows,
          "m_vsm [m/s^2] and m_hsm [rad/s^2] are the paper's measures (p. 22);"
          " both are MINIMISED. stop%/revers/path* are context, not from the paper.")

    real_gammas = {g for g in gammas if g is not None}
    if len(real_gammas) > 1 or (real_gammas and None in gammas):
        print("\n  WARNING: these groups do not share one discount, so the discounted")
        print("  returns above are NOT comparable across them. Use undiscounted.")
        print(f"  gammaPerSecond seen: {sorted(real_gammas)}"
              + (" plus runs predating the column" if None in gammas else ""))
    real_ts = {t for t in ts_values if t is not None}
    if len(real_ts) > 1:
        print(f"\n  WARNING: mixed control periods {sorted(real_ts)} - 'steps' and")
        print("  'totalS' are not comparable across these groups.")

    if show_runs:
        print("\nPER-RUN")
        for k in keys:
            print(f"\n  {group_label(k)}")
            for r in sorted(groups[k], key=lambda x: int(x.exp_num)):
                print(f"    {r.exp_num:>3}  goal={bool(r.get('reachGoal', False))!s:5} "
                      f"volColl={bool(r.get('collision', False))!s:5} "
                      f"obsColl={bool(r.get('Obscollision', False))!s:5} "
                      f"steps={fmt(r.get('nSteps'), '.0f'):>5} "
                      f"undisc={fmt(r.get('undiscountedReturn'))}")

    if csv_out:
        pd.DataFrame(wide).to_csv(csv_out, index=False)
        print(f"\nwrote {csv_out}  ({len(wide)} groups x {len(wide[0]) if wide else 0} columns)")
    if runs_csv_out:
        pd.DataFrame([{
            "source": r.source, "trajectories": r.trajectories,
            "algorithm": r.algorithm, "suffix": r.suffix, "expNum": int(r.exp_num),
            "outcome": outcome_of(r),
            "reachGoal": bool(r.get("reachGoal", False)),
            "collision": bool(r.get("collision", False)),
            "Obscollision": bool(r.get("Obscollision", False)),
            "maxSteps": bool(r.get("maxSteps", False)),
            "nSteps": r.get("nSteps"),
            "discountedReturn": r.get("discountedReturn"),
            "undiscountedReturn": r.get("undiscountedReturn"),
            "simNum": r.get("simNum"),
            "meanTreeDepth": r.get("meanTreeDepth"),
            "maxTreeDepth": r.get("maxTreeDepth"),
            "meanRolloutDepth": r.get("meanRolloutDepth"),
            "maxRolloutDepth": r.get("maxRolloutDepth"),
            "ts": r.get("ts"), "radiusScale": r.get("radiusScale"),
            "explorationC": r.get("explorationC"),
            "gammaPerSecond": r.get("gammaPerSecond"),
            **{k: v for k, v in
               zip(("m_vsm", "m_hsm", "stop_pct", "reversals", "path_len", "path_eff"),
                   (smoothness([r], legacy_ts)[k] for k in
                    ("m_vsm", "m_hsm", "stop_pct", "reversals", "path_len", "path_eff")))},
        } for r in sorted(runs, key=lambda x: (group_key(x), int(x.exp_num)))]
        ).to_csv(runs_csv_out, index=False)
        print(f"wrote {runs_csv_out}  ({len(runs)} runs, one row each)")


# --------------------------------------------------------------- plotting --

def _plot_setup():
    """Import matplotlib headless, plus the project's own frame renderer.

    plot_frame2 lives in MCTS_VO/experiment_utils.py and is the same function
    debug_utils uses for the end-of-run animation; reusing it means the picture
    here is the picture you already know, rather than a second dialect of it.
    """
    import matplotlib
    matplotlib.use("Agg")                       # no display when run over ssh
    import matplotlib.pyplot as plt
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from MCTS_VO.experiment_utils import plot_frame2, plot_robot
    return plt, plot_frame2, plot_robot


class _Config:
    """The two fields plot_robot reads, without constructing a whole EnvConfig."""

    def __init__(self, robot_radius=0.15):
        self.robot_radius = robot_radius


# Outcome buckets, in precedence order: a run is filed under the first that
# applies. "collision" is the voluntary one - the robot drove into something,
# which is what VO exists to prevent - and is kept distinct from obsCollision,
# where an obstacle hit a robot that had already stopped.
OUTCOMES = ("success", "collision", "obsCollision", "timeout")
OUTCOME_COLOUR = {"success": "tab:green", "collision": "tab:red",
                  "obsCollision": "tab:orange", "timeout": "tab:gray"}


def outcome_of(run):
    if bool(run.get("reachGoal", False)):
        return "success"
    if bool(run.get("collision", False)):
        return "collision"
    if bool(run.get("Obscollision", False)):
        return "obsCollision"
    return "timeout"


def plot_dir(base, run, outcome=None, flat=False):
    """<base>/[<snapshot>/]<algorithm>/<environment>/<outcome>/

    Sorting the images this way means "show me every voluntary collision
    VO-TREE had on the sinusoidal scene" is a directory listing rather than a
    search, which is the usual question when a batch has just finished.

    `flat` drops the snapshot level, for plots written inside the snapshot's own
    archive directory, where repeating the label would be redundant.
    """
    parts = [base]
    if run.source and not flat:
        parts.append(run.source)
    parts += [f"{run.algorithm}{run.suffix}", run.trajectories]
    if outcome:
        parts.append(outcome)
    return os.path.join(*parts)


def check_goal(runs, goal):
    """Warn if the assumed goal disagrees with where successful runs stopped.

    Cheap insurance against plotting a whole batch against the wrong target -
    which is exactly the defect the goal-coordinate fix addressed, and it is
    invisible in a picture unless something says so.
    """
    ends = []
    for r in runs:
        if not bool(r.get("reachGoal", False)):
            continue
        trj = r.aux("trj_")
        if trj is not None and len(trj):
            ends.append(np.asarray(trj, dtype=float)[-1, :2])
    if not ends:
        return
    d = float(np.median(np.linalg.norm(np.array(ends) - np.array(goal), axis=1)))
    if d > 0.30:
        warn(f"assumed goal {tuple(goal)} is {d:.2f} m from where the "
             f"{len(ends)} successful runs actually stopped - wrong --goal?")


def plot_run(run, goal, out, animate=False, fps=10):
    """Render one run: either a still overview or the frame-by-frame animation."""
    plt, plot_frame2, plot_robot = _plot_setup()

    trj = run.aux("trj_")
    if trj is None:
        warn(f"{run.algorithm}_{run.exp_num}{run.suffix}: no trj_ pickle, skipped")
        return None
    trj = np.asarray(trj, dtype=float)
    cfg = _Config()
    # obs_ is optional: archives made before it was included in the snapshot
    # have the path but not the perception, and a path-only plot is far better
    # than no plot at all.
    obs = run.aux("obs_")
    n = len(trj) if obs is None else min(len(trj), len(obs))

    if animate:
        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        # plot_frame2 indexes points_list per frame; the raw LIDAR points are
        # not persisted by save_data, so pass an empty set per frame. Everything
        # else it draws - robot, goal, detected obstacles, path - is on disk.
        empty_points = [()] * n
        # plot_frame2 unpacks obs[i] per frame, so an archive without obs_ needs
        # an empty pair per frame rather than None - same tolerance the still
        # plot has, and it keeps the path visible when the perception is gone.
        frames_obs = obs if obs is not None else [(np.empty((0, 4)), np.empty(0))] * n
        ani = FuncAnimation(
            fig, plot_frame2,
            fargs=(goal, cfg, frames_obs, trj, ax,
                   (np.empty((0, 4)), np.empty(0)), empty_points),
            frames=range(n), save_count=None, cache_frame_data=False)
        ani.save(out, fps=fps)
        plt.close(fig)
        return out

    # Still overview: the whole run in one image, which is what you want when
    # skimming thirty of them.
    fig, ax = plt.subplots(figsize=(6, 6))
    if obs is not None:
        seen = np.vstack([np.asarray(o[0], dtype=float)[:, :2]
                          for o in obs[:n] if len(o[0])]) if n else np.empty((0, 2))
        if len(seen):
            ax.plot(seen[:, 0], seen[:, 1], ".", color="0.75", ms=2,
                    label="obstacles seen (all steps)")
        last_x, last_rad = obs[n - 1]
        for c, rr in zip(np.asarray(last_x, dtype=float),
                         np.asarray(last_rad, dtype=float)):
            ax.add_artist(plt.Circle((c[0], c[1]), rr, color="k", alpha=0.35))
    ax.plot(trj[:n, 0], trj[:n, 1], "--r", lw=1, label="path")
    ax.plot(trj[0, 0], trj[0, 1], "o", color="tab:green", label="start")
    ax.plot(goal[0], goal[1], "xb", ms=12, mew=3, label="goal")
    plot_robot(trj[n - 1, 0], trj[n - 1, 1], trj[n - 1, 2], cfg, ax)

    ax.set_title(f"{run.algorithm}_{run.exp_num}{run.suffix}  ({run.trajectories})\n"
                 f"{outcome_of(run)}, {n} steps"
                 + ("" if obs is not None else "  [no obs_ data archived]"),
                 fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-4, 2)
    ax.set_ylim(-4, 2)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_grid(runs, goal, out, cols=6):
    """Every run of a group as one sheet of small path plots."""
    plt, _, _ = _plot_setup()
    runs = sorted(runs, key=lambda r: int(r.exp_num))
    rows = int(math.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows),
                             squeeze=False)
    for ax, r in zip([a for row in axes for a in row], runs):
        trj = r.aux("trj_")
        if trj is None:
            ax.axis("off")
            continue
        t = np.asarray(trj, dtype=float)
        kind = outcome_of(r)
        ax.plot(t[:, 0], t[:, 1], "-", color=OUTCOME_COLOUR[kind], lw=1)
        ax.plot(goal[0], goal[1], "xb", ms=6, mew=2)
        ax.plot(t[0, 0], t[0, 1], ".", color="k", ms=4)
        ax.set_title(f"{r.exp_num}: {kind}", fontsize=7)
        ax.set_xlim(-4, 2); ax.set_ylim(-4, 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in [a for row in axes for a in row][len(runs):]:
        ax.axis("off")
    tally = {o: sum(1 for r in runs if outcome_of(r) == o) for o in OUTCOMES}
    fig.suptitle(f"{runs[0].algorithm}{runs[0].suffix} / {runs[0].trajectories}"
                 f"  -  {len(runs)} runs  ("
                 + ", ".join(f"{o} {tally[o]}" for o in OUTCOMES if tally[o])
                 + ")  green=success, red=collision, orange=obsCollision, grey=timeout",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def do_plot(args, runs):
    if not runs:
        sys.exit("no runs found to plot")
    goal = tuple(args.goal)
    sel = [r for r in runs
           if f"{r.algorithm}_{r.exp_num}{r.suffix}".startswith(args.run)
           or f"{r.algorithm}{r.suffix}" == args.run]
    if args.scene:
        sel = [r for r in sel if r.trajectories == args.scene]
    if not sel:
        sys.exit(f"nothing matches '{args.run}'"
                 + (f" in scene {args.scene}" if args.scene else ""))
    check_goal(sel, goal)

    if args.grid:
        # A grid spans every outcome, so it belongs one level above them.
        by = defaultdict(list)
        for r in sel:
            by[plot_dir(args.outdir, r)].append(r)
        for d, g in sorted(by.items()):
            os.makedirs(d, exist_ok=True)
            r0 = g[0]
            out = os.path.join(d, f"grid_{r0.algorithm}{r0.suffix}_{r0.trajectories}.png")
            print(f"  {rel(plot_grid(g, goal, out), args.outdir)}  ({len(g)} runs)")
        return

    written = defaultdict(int)
    ordered = sorted(sel, key=lambda x: (group_key(x), int(x.exp_num)))
    if args.limit:
        ordered = ordered[:args.limit]
    if args.anim and len(ordered) > 20:
        # ~0.04 s per simulation step, measured over runs of 122 and 335 steps.
        secs = sum(float(r.get("nSteps", 200) or 200) for r in ordered) * 0.04
        warn(f"{len(ordered)} animations, roughly {secs / 60:.0f} min; "
             f"--limit caps it and Ctrl-C is safe (stills are already written)")

    # Two passes, not one interleaved pass: the stills take seconds for a whole
    # campaign and the animations take tens of minutes, so finishing every still
    # first means interrupting the slow half still leaves a complete set.
    stems = []
    for r in ordered:
        d = plot_dir(args.outdir, r, outcome_of(r))
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, f"{r.algorithm}_{r.exp_num}{r.suffix}")
        stems.append((r, d, stem))
        if plot_run(r, goal, stem + ".png"):
            written[d] += 1

    if args.anim:
        for i, (r, d, stem) in enumerate(stems, 1):
            # Progress matters here: a silent process for half an hour is
            # indistinguishable from a hung one.
            print(f"  [{i}/{len(stems)}] animating "
                  f"{r.algorithm}_{r.exp_num}{r.suffix} ({r.trajectories})...",
                  flush=True)
            if plot_run(r, goal, stem + ".gif", animate=True, fps=args.fps):
                written[d] += 1

    # The grid is part of "draw this group", so it comes along automatically
    # rather than needing a second command with --grid.
    if not args.no_grid:
        by = defaultdict(list)
        for r in ordered:
            by[plot_dir(args.outdir, r)].append(r)
        for d, g in by.items():
            os.makedirs(d, exist_ok=True)
            r0 = g[0]
            plot_grid(g, goal, os.path.join(
                d, f"grid_{r0.algorithm}{r0.suffix}_{r0.trajectories}.png"))
            written[d] += 1

    for d in sorted(written):
        print(f"  {rel(d, args.outdir)}/  {written[d]} file(s)")
    if not written:
        print("  nothing written")


# ----------------------------------------------------------------- snapshot --

def _human(nbytes):
    for unit in ("B", "kB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def snapshot(src, label, archive, force=False, keep=False,
             animations=False, goal=DEFAULT_GOAL, make_plots=True):
    """Freeze a debug folder under a label, then clear the original.

    The order is deliberate: copy everything, verify every copy against its
    source, draw the plots, and only then delete - and delete *only* the files
    that were verified into the archive. Anything not archived (the animations,
    unless asked for) stays where it is rather than being quietly lost.
    """
    dest = os.path.join(archive, label)
    if os.path.exists(dest):
        if not force:
            sys.exit(f"{dest} already exists; pass --force to replace it")
        shutil.rmtree(dest)
    if not os.path.isdir(src):
        sys.exit(f"{src} does not exist")

    copied = []          # (source, destination) pairs that verified
    for dirpath, dirnames, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        if not animations and os.path.basename(dirpath) == ANIMATION_DIR:
            dirnames[:] = []
            continue
        in_anim = ANIMATION_DIR in rel.split(os.sep)
        for f in files:
            if not in_anim and not f.startswith(SNAPSHOT_PREFIXES):
                continue
            out = os.path.join(dest, rel) if rel != "." else dest
            os.makedirs(out, exist_ok=True)
            s, d = os.path.join(dirpath, f), os.path.join(out, f)
            shutil.copy2(s, d)
            if os.path.getsize(s) != os.path.getsize(d):
                sys.exit(f"copy of {f} does not match its source; nothing deleted")
            copied.append((s, d))

    if not copied:
        sys.exit(f"no run artefacts found under {src}")
    total = sum(os.path.getsize(d) for _, d in copied)
    print(f"archived {len(copied)} files ({_human(total)}) -> {dest}")

    if make_plots:
        runs = discover(dest, label=label)
        if runs:
            print("drawing plots into the archive...")
            n = plot_archive(runs, dest, goal)
            print(f"  {n} image(s) -> {os.path.join(dest, 'plots')}")

    if keep:
        print(f"--keep given: {src} left untouched")
        return

    for s, _ in copied:
        os.remove(s)
    # Tidy up directories the deletion emptied, so debug/ is ready for the next
    # batch rather than a husk of empty scene folders.
    left = sum(len(files) for _, _, files in os.walk(src))
    for dirpath, _, _ in os.walk(src, topdown=False):
        if dirpath != src and not os.listdir(dirpath):
            os.rmdir(dirpath)
    print(f"removed {len(copied)} archived files from {src}"
          + (f"; {left} file(s) left behind (not archived)" if left else ""))


def plot_archive(runs, dest, goal):
    """Draw every archived run into <archive>/<label>/plots/."""
    base = os.path.join(dest, "plots")
    n = 0
    for r in runs:
        d = plot_dir(base, r, outcome_of(r), flat=True)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"{r.algorithm}_{r.exp_num}{r.suffix}.png")
        if plot_run(r, goal, out):
            n += 1
    by = defaultdict(list)
    for r in runs:
        by[plot_dir(base, r, flat=True)].append(r)
    for d, g in by.items():
        os.makedirs(d, exist_ok=True)
        r0 = g[0]
        if plot_grid(g, goal,
                     os.path.join(d, f"grid_{r0.algorithm}{r0.suffix}_{r0.trajectories}.png")):
            n += 1
    return n


# --------------------------------------------------------------------- main --

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Summarize MCTS-VO experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  %(prog)s                       summarize debug/ and every snapshot\n"
               "  %(prog)s --dir ~/.local/share/Trash/files/debug\n"
               "  %(prog)s --runs --csv out.csv\n"
               "  %(prog)s snapshot B2           freeze debug/ as debug_archive/B2\n"
               "  %(prog)s plot VO-TREE_3        one run, as a still\n"
               "  %(prog)s plot VO-TREE --grid   every VO-TREE run on one sheet\n"
               "  %(prog)s plot VO-TREE_3 --anim animation, using the project's\n"
               "                                 own plot_frame2\n")
    p.add_argument("--dir", default=os.path.join(here, "debug"),
                   help="debug folder to read (default: %(default)s)")
    p.add_argument("--archive", default=os.path.join(here, "debug_archive"),
                   help="where snapshots live (default: %(default)s)")
    # Written every run rather than on request: the tables scroll away, and the
    # point of the summary is not having to re-run it to look something up.
    p.add_argument("--csv", default=os.path.join(here, "debug_summary.csv"),
                   help="one row per group (default: %(default)s)")
    p.add_argument("--runs-csv", default=os.path.join(here, "debug_runs.csv"),
                   help="one row per individual run (default: %(default)s)")
    p.add_argument("--no-csv", action="store_true",
                   help="print the tables only, write nothing")
    p.add_argument("--runs", action="store_true", help="list individual runs too")
    p.add_argument("--legacy-ts", type=float, default=0.1,
                   help="control step assumed for runs with no ts column "
                        "(default: %(default)s)")
    p.add_argument("--no-archive", action="store_true",
                   help="summarize only --dir, ignoring snapshots")
    sub = p.add_subparsers(dest="cmd")

    def shared(sp):
        """Let --dir/--archive be given after the subcommand as well as before.

        SUPPRESS is the point: without it, a subparser option that was not
        given writes None into the namespace and silently overrides the value
        the top-level parser already put there.
        """
        sp.add_argument("--dir", default=argparse.SUPPRESS)
        sp.add_argument("--archive", default=argparse.SUPPRESS)
        sp.add_argument("--no-archive", action="store_true",
                        default=argparse.SUPPRESS)
        return sp

    sp = sub.add_parser("snapshot",
                        help="archive a debug folder under a label, plot it, "
                             "and clear the original")
    sp.add_argument("label")
    sp.add_argument("--force", action="store_true",
                    help="replace an existing label")
    sp.add_argument("--keep", action="store_true",
                    help="copy instead of move: leave the debug folder as it is")
    sp.add_argument("--with-animations", action="store_true",
                    help="archive the rendered GIFs and MP4s too. They are "
                         "~80%% of the folder and regenerable, so they are "
                         "normally left in place rather than copied")
    sp.add_argument("--no-plots", action="store_true",
                    help="archive without drawing the plots")
    sp.add_argument("--goal", type=float, nargs=2, default=list(DEFAULT_GOAL),
                    metavar=("X", "Y"))
    shared(sp)

    pp = sub.add_parser("plot", help="draw what the robot saw and where it went")
    pp.add_argument("run", nargs="?", default="",
                    help="run id (VO-TREE_3) or a whole group (VO-TREE). "
                         "Omit it to draw everything.")
    pp.add_argument("--scene", help="restrict to sinusoidal or intention")
    pp.add_argument("--grid", action="store_true",
                    help="draw ONLY the group sheets, no per-run images")
    pp.add_argument("--no-grid", action="store_true",
                    help="skip the group sheets")
    pp.add_argument("--anim", action="store_true",
                    help="also write animations, using the project's "
                         "plot_frame2. Measured at about 0.04 s per simulation "
                         "step, so 6-15 s per run and roughly half an hour for "
                         "a full 180-run campaign, against 15 s for the stills.")
    pp.add_argument("--fps", type=int, default=10)
    pp.add_argument("--limit", type=int, default=0,
                    help="cap how many runs are drawn (0 = all). Worth setting "
                         "with --anim, which takes seconds per run rather than "
                         "a fraction of one.")
    pp.add_argument("--goal", type=float, nargs=2, default=list(DEFAULT_GOAL),
                    metavar=("X", "Y"),
                    help="goal position, which runs do not record "
                         "(default: %(default)s)")
    pp.add_argument("--outdir", default=os.path.join(here, "debug_plots"))
    shared(pp)
    args = p.parse_args()

    if args.cmd == "snapshot":
        snapshot(args.dir, args.label, args.archive, force=args.force,
                 keep=args.keep, animations=args.with_animations,
                 goal=tuple(args.goal), make_plots=not args.no_plots)
        return

    runs = discover(args.dir)
    if not runs:
        print(f"no runs found in {args.dir}")
    if not args.no_archive and os.path.isdir(args.archive):
        for label in sorted(os.listdir(args.archive)):
            d = os.path.join(args.archive, label)
            if os.path.isdir(d):
                runs += discover(d, label=label)
    if not runs:
        sys.exit(1)
    if args.cmd == "plot":
        do_plot(args, runs)
        return
    summarize(runs, args.legacy_ts,
              None if args.no_csv else args.csv, args.runs,
              None if args.no_csv else args.runs_csv)


if __name__ == "__main__":
    main()
