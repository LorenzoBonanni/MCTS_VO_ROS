# summarize_debug.py

Reads the run artefacts in `debug/` and prints five tables: what happened, how well it scored,
how long it took, how deep it searched, and how smooth the motion was. It also snapshots a
`debug/` folder under a label, so that re-running after each bug fix does not silently overwrite
the previous results.

Both this file and the script are **untracked on purpose**. They sit next to the code but are
never part of a commit, so `git checkout` and `git cherry-pick` never touch them, and stepping
through the bug-fix commits does not keep deleting your tooling.

## Quick start

The script needs `pandas`, which the system Python does not have, so source the venv first:

```bash
cd ~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos
source ../venv/bin/activate

python3 summarize_debug.py
```

That reads `debug/` plus every snapshot in `debug_archive/`, prints the tables, **and writes two
CSVs** so you never have to re-run it just to look something up:

| file | contents |
|---|---|
| `analysis/summary.csv` | one row per group — the tables above, in one wide row each |
| `analysis/runs.csv` | one row per individual run, including its outcome and smoothness |

They sit in `analysis/` rather than beside the script because they are generated, and rather than
in `debug/` or `debug_archive/` because they summarise across both — a snapshot clears `debug/`,
and these have to outlive that.

Open them in a spreadsheet, or:

```python
import pandas as pd
d = pd.read_csv("analysis/runs.csv")
d.groupby(["algorithm", "trajectories"]).outcome.value_counts().unstack(fill_value=0)
```

Other things you may want:

```bash
# only the live folder, ignoring snapshots
python3 summarize_debug.py --no-archive

# the older experiments, which are sitting in the trash
python3 summarize_debug.py --dir ~/.local/share/Trash/files/debug

# list every run individually in the terminal too
python3 summarize_debug.py --runs

# write the CSVs somewhere else, or not at all
python3 summarize_debug.py --csv mine.csv --runs-csv mine_runs.csv
python3 summarize_debug.py --no-csv          # print the tables only
```

## Comparing the bug fixes step by step

Every run writes `data_<algo>_<n>.csv`. There is no run id in the name, so the next batch
overwrites the last one. Snapshot before moving on:

```bash
# ... run your experiments at B1 ...
python3 summarize_debug.py snapshot B1

git cherry-pick 43010d5          # B2: gt_obs_pos
# ... run again ...
python3 summarize_debug.py snapshot B2

git cherry-pick 4b18117          # B3: max_obs_vel
# ... and so on
```

`snapshot` **moves rather than copies**. It archives every run artefact into
`debug_archive/<label>/`, draws the plots into `debug_archive/<label>/plots/`, and then clears
`debug/` so the next batch starts on an empty folder. A snapshot is self-contained: data and
pictures for that step, in one directory.

The order is deliberate — copy, verify every file against its source, plot, and only then delete.
It deletes **only what it verified into the archive**, so nothing can be lost to a partial copy.

```bash
python3 summarize_debug.py snapshot B2 --keep              # copy, don't clear debug/
python3 summarize_debug.py snapshot B2 --with-animations   # archive the GIFs too
python3 summarize_debug.py snapshot B2 --no-plots          # archive without drawing
python3 summarize_debug.py snapshot B2 --force             # replace an existing label
```

The rendered `animations/` folder is **not** archived by default: it is about 80 % of the folder
(210 MB of 260 MB on a full campaign) and can be regenerated from the pickles. Because it is not
archived it is also not deleted — it stays in `debug/`, and the script says so. Everything else,
roughly 50 MB per campaign, is archived.

Then `python3 summarize_debug.py` shows every snapshot next to the live folder, prefixed with its
label, so you can watch the numbers move as each fix lands.

## Pictures

```bash
python3 summarize_debug.py plot                     # EVERYTHING: stills + group sheets
python3 summarize_debug.py plot --anim              # ...and animations too
python3 summarize_debug.py plot VO-TREE             # every VO-TREE run, both scenes
python3 summarize_debug.py plot VO-TREE_3           # a single run
python3 summarize_debug.py plot VO-TREE --grid      # only the group sheet
```

**Recreating everything after deleting `debug_plots/`** — the run argument is optional, and the
sources are `debug/` plus every snapshot, so one command rebuilds the lot:

```bash
python3 summarize_debug.py plot            # ~15 s for a 180-run campaign
python3 summarize_debug.py plot --anim     # ~20 min, animations included
```

Timings are measured: stills are a fraction of a second each, animations about 0.04 s per
simulation step, so 6–15 s per run depending on its length. `--limit N` caps it, and Ctrl-C is
safe — the stills are written before the animations start.

To rebuild the plots *inside* a snapshot rather than in `debug_plots/`:

```bash
python3 summarize_debug.py plot --dir debug_archive/B1 --no-archive \
                                --outdir debug_archive/B1/plots
```

Images are filed by **algorithm → environment → outcome**, so finding "every voluntary collision
VO-TREE had on the sinusoidal scene" is a directory listing rather than a search:

```
debug_plots/
  VO-TREE/
    sinusoidal/
      grid_VO-TREE_sinusoidal.png     <- the whole group on one sheet
      success/        VO-TREE_0.png  VO-TREE_11.png  ...   (17)
      collision/      VO-TREE_14.png                       (1)
      obsCollision/   VO-TREE_1.png   VO-TREE_2.png  ...   (12)
    intention/
      success/ collision/ obsCollision/ timeout/
  MCTS/ ...
  VO-PLANNER/ ...
```

The four outcome folders are `success`, `collision` (**voluntary** — the robot drove into
something, the case VO must prevent), `obsCollision` (an obstacle hit a robot that had already
stopped) and `timeout`. Only outcomes that actually happened get a folder, so an empty-looking
tree is a good sign: `VO-PLANNER/intention/` contains nothing but `success/` because it reached
the goal in all 30 runs.

The grid sits one level up, next to the outcome folders, because it spans all of them.

Plots for a **snapshot** live inside that snapshot instead, so each archived step is
self-contained:

```
debug_archive/B1/
  sinusoidal/  intention/          <- the run artefacts
  plots/VO-TREE/sinusoidal/success/...
```

`snapshot` draws them for you. Re-running `plot` later writes into `debug_plots/<label>/...`
rather than back into the archive, so the archive is never modified after the fact.

Plotting a whole group draws every run by default. `--limit N` caps it, which mainly matters with
`--anim` — animations take seconds each, stills a fraction of one.

**`--grid`** is the one to reach for first: every run of a group as a small path plot, coloured by
outcome — green success, red voluntary collision, orange obstacle collision, grey timeout — with
the tally in the title. Thirty runs on one sheet makes the failure pattern obvious in a way the
percentages do not. On the current data you can see at a glance that the obstacle collisions all
end in the same stretch of the map.

**A single run** draws the path, the start, the goal, the robot at its final pose, every obstacle
detection over the whole run as faint grey dots, and the last step's detections as filled circles.
Those circles are the radii *as VO saw them*, i.e. after `RADIUS_SCALE`, so if they look far bigger
than the 0.1 m obstacles really are, that is the scale factor and not a bug in the plot.

**`--anim`** calls `plot_frame2` from `MCTS_VO/experiment_utils.py` — the same renderer
`debug_utils.py` uses at the end of a run, so the animation is the one you already know. It
*adds* animations rather than replacing the stills, and prints per-run progress, since a full
campaign runs for twenty minutes and a silent process that long is hard to tell from a hung one.

Three things worth knowing:

- **`debug_archive/B1` was made before `obs_` was archived**, so its plots show the path, start,
  goal and final pose but no detected obstacles. The title says `[no obs_ data archived]` when
  that happens. The perception data for that batch is gone — `debug/` was cleared after the
  snapshot was taken. Every snapshot from now on includes it.
- **The raw LIDAR points are not saved.** `save_data` never pickles `points_list`, so the green
  scan dots you get from a live run cannot be reproduced afterwards. What the plots show instead is
  the *fitted obstacles*, which is what the planner actually consumed. If you want the raw points
  in offline plots, `points_list` has to be added to `save_data`.
- **The goal is not saved either**, so the plot has to be told. It defaults to the corrected
  `(-2.783, -0.720)`; pass `--goal X Y` for anything else. As a safety net the command checks the
  assumed goal against where the successful runs actually stopped and warns if they disagree —
  which is exactly what a wrong goal coordinate looks like:

  ```
  ! assumed goal (-3.26, -1.61) is 1.11 m from where the 1 successful runs
    actually stopped - wrong --goal?
  ```

## What the tables mean

### OUTCOMES

```
                  group   n  goal%  volColl%  obsColl%  timeout%  steps
  ---------------------  --  -----  --------  --------  --------  -----
   intention/VO-PLANNER  30    100         0         0         0    129
      intention/VO-TREE  30     50         0        37        13    259
        sinusoidal/MCTS  30      3        80        17         0    148
```

- **goal%** — reached the goal. Higher is better.
- **volColl%** — *voluntary* collision: the robot was moving and drove into something. This is the
  number velocity obstacles exist to keep at zero, so it is the one to watch. Plain `MCTS` has no
  VO and sits at 70–80%; both VO variants are at 0–3%.
- **obsColl%** — *involuntary*: an obstacle moved into a robot that was stopped. Not something the
  planner can prevent, so do not read it as a planner failure.
- **timeout%** — ran out of steps without reaching the goal or crashing.
- **steps** — mean planning steps per run.

Read the row above like this: `VO-TREE` on the intention scene never drove into anything, but only
got to the goal half the time, and 37% of runs ended with something hitting it while it waited.

### RETURNS

Mean ± standard deviation of the discounted and undiscounted return. **See the pitfalls below
before comparing discounted returns between groups.**

### TIMING

Milliseconds. `senseMed`/`senseP99` is obstacle estimation, `planMed` is the planner, `cycleMed` is
the whole control period measured command-to-command, `Hz` is `1/cycleMed`, `totalS` is the mean
wall-clock length of a run, `sims/step` is how many simulations the planner got through.

A group marked with `*` had its timing **reconstructed, not measured** — see pitfalls.

`planMed` is a budget, not a speed: the planner runs until its time is up, so it always equals
whatever it was given. Making the planner faster raises `sims/step`, not lower `planMed`.

### DEPTH

Search tree depth and rollout length, mean and max. Blank for `VO-PLANNER`, which has no tree at
all — that is correct, not missing data.

### SMOOTHNESS

```
                  group  m_vsm   m_hsm  stop%  revers  pathLen  pathEff
  ---------------------  -----  ------  -----  ------  -------  -------
   intention/VO-PLANNER  0.066   0.613    0.0     0.0     3.12    1.000
      intention/VO-TREE  0.819  17.633   17.3    31.9     3.38    0.807
```

The first two are the paper's measures (ROBOT-D-25-00227, p. 22). **Both are minimised.**

**`m_vsm`** — linear velocity smoothness, in m/s². The mean change in the velocity *vector*
between consecutive commands, divided by the control step:

> m_vsm = 1/(T−1) · Σ ‖**v**₍t+1₎ − **v**₍t₎‖₂ / t_s,  where **v** = ⟨v·cos α, v·sin α⟩

Because it uses the vector rather than the speed, turning sharply at constant speed still counts
as unsmooth — which is the point.

**`m_hsm`** — heading smoothness, in rad/s². The same idea on the angular velocity commands:

> m_hsm = 1/(T−1) · Σ |ω₍t+1₎ − ω₍t₎| / t_s

The remaining four are **not from the paper**. They are there because a bad smoothness number on
its own does not tell you why:

- **stop%** — share of commands with zero speed. A robot that repeatedly halts and restarts scores
  badly on `m_vsm` without ever steering erratically, and that is the usual explanation when
  `VO-TREE` looks rough: 17–19% stopped, against 0–3% for `VO-PLANNER`.
- **revers** — how many times the commanded speed changed sign, i.e. how often VO forced reverse.
- **pathLen** — metres travelled.
- **pathEff** — straight-line distance from start to finish ÷ path length, so 1.0 is a straight
  line. `VO-PLANNER` at 1.000 went directly there; `VO-TREE` at 0.807 wandered.

## Things that will mislead you

**Discounted return is not comparable across the B5 fix.** The discount changed there from 0.9 per
step to 0.81 per second (0.979 per step), and each run's return was computed with whatever was
active at the time. Comparing discounted returns across that boundary is meaningless. Undiscounted
return is unaffected. The script prints a warning when a summary spans groups with different
discounts.

**`steps` and `totalS` are not comparable across a change in `ts`.** A step is 0.1 s in the default
configuration and 0.02 s at 25 Hz, so the step count changes even when nothing about the behaviour
does. Another warning covers this.

**Timing rows marked `*` were reconstructed, not measured.** Runs made before the `step_stats`
instrumentation have no per-phase record, but the split is still exactly recoverable, because the
loop stored `times[i] = ts − t_sense − 0.005`. So `t_sense` and `t_plan` are real numbers, not
guesses — but `cycleMed` for those rows is the configured period rather than an observed one, and
`sims/step` comes from a separate file. Once the instrumentation commit is on your branch the
asterisk disappears and everything is measured. If a run has no `ts` column the script assumes
0.1 s; override with `--legacy-ts`.

**Blank cells are usually correct.** `VO-PLANNER` has no tree, so depth and `sims/step` are empty
for it. A dash means "this run never recorded that", not "zero".

**A group is one configuration, not one experiment.** Runs are grouped by
`(snapshot, scene, algorithm, filename suffix)`. Two batches run at different settings but with the
same filenames land in the same group and get averaged together — which is exactly why the
snapshot command exists.

## Where things live

| | |
|---|---|
| `summarize_debug.py` | the script (untracked) |
| `SUMMARIZE_README.md` | this file (untracked) |
| `debug/` | live results, gitignored |
| `debug_archive/<label>/` | snapshots: data + `plots/` (untracked) |
| `debug_plots/<algo>/<env>/<outcome>/` | output of `plot` (untracked) |
| `analysis/summary.csv` | one row per group, rewritten every run (untracked) |
| `analysis/runs.csv` | one row per run, rewritten every run (untracked) |
| `~/.local/share/Trash/files/debug/` | earlier experiments: RADIUS_SCALE sweep, 5/10/20/25 Hz sweep |

`--dir` reads any of these in place. Nothing is ever moved or deleted; `snapshot` copies.
