# summarize_debug.py

Reads the run artefacts in `debug/` and prints five tables: what happened, how well it scored,
how long it took, how deep it searched, and how smooth the motion was. It also snapshots a
`debug/` folder under a label, so that re-running does not silently overwrite the previous results.

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

# narrow it down: any combination of the three, each a comma-separated list
python3 summarize_debug.py --algo VO-TREE            # one algorithm, every snapshot
python3 summarize_debug.py --algo MCTS,VO-TREE       # two, side by side
python3 summarize_debug.py --scene intention         # one environment
python3 summarize_debug.py --label B3                # one snapshot
python3 summarize_debug.py --label B3,B4             # compare two steps
python3 summarize_debug.py --label ""                # the live folder only
python3 summarize_debug.py --algo MCTS,VO-TREE --scene sinusoidal --label B1

# the older experiments, which are sitting in the trash
python3 summarize_debug.py --dir ~/.local/share/Trash/files/debug

# list every run individually in the terminal too
python3 summarize_debug.py --runs

# write the CSVs somewhere else, or not at all
python3 summarize_debug.py --csv mine.csv --runs-csv mine_runs.csv
python3 summarize_debug.py --no-csv          # print the tables only
```

## Comparing several batches

Every run writes `data_<algo>_<n>.csv`. There is no run id in the name, so **the next batch
overwrites the last one**. Snapshot before changing anything and running again:

```bash
# ... run a batch ...
python3 summarize_debug.py snapshot B4

# ... change something, run again ...
python3 summarize_debug.py snapshot B5

# ... and so on
```

The label is any name you like — `B4`, `rs18`, `fast-obstacles`. Then a plain
`python3 summarize_debug.py` shows every snapshot next to the live folder, each row prefixed with
its label, so you can watch the numbers move from one batch to the next.

`snapshot` **moves rather than copies**. It archives every run artefact into
`debug_archive/<label>/`, draws the plots into `debug_archive/<label>/plots/`, and then clears
`debug/` so the next batch starts on an empty folder. A snapshot is self-contained: data and
pictures for that step, in one directory.

The order is deliberate — copy, verify every file against its source, plot, and only then delete.
It deletes **only what it verified into the archive**, so nothing can be lost to a partial copy.

```bash
python3 summarize_debug.py snapshot B2 --keep              # copy, don't clear debug/
python3 summarize_debug.py snapshot B2 --with-animations   # render animations into it too
python3 summarize_debug.py snapshot B2 --no-plots          # archive without drawing
python3 summarize_debug.py snapshot B2 --force             # replace an existing label
```

**Two different sets of animations exist, and the archive holds only one of them.** The simulator
writes its own per-run GIFs and MP4s into `debug/<scene>/animations/` at the end of each run.
Those are *never* archived and *never* deleted — they stay where they are, and the script says how
many it left behind. The pictures that go into the archive are the ones this script renders, into
`<label>/plots/<algorithm>/<scene>/<outcome>/`, so they match the stills and the outcome
foldering. `--with-animations` renders those; it does not copy anything.

Then `python3 summarize_debug.py` shows every snapshot next to the live folder, prefixed with its
label, so you can watch the numbers move as each fix lands.

## Pictures

```bash
python3 summarize_debug.py plot                     # EVERYTHING: stills + group sheets
python3 summarize_debug.py plot --anim              # ...and animations too
python3 summarize_debug.py plot VO-TREE             # every VO-TREE run, both scenes
python3 summarize_debug.py plot VO-TREE_3           # a single run
python3 summarize_debug.py plot VO-TREE --grid      # only the group sheet
python3 summarize_debug.py plot --label B3 --anim   # one snapshot only
```

`--algo`, `--scene` and `--label` work on **both** the summary and `plot`, and can be given
before or after the subcommand. They combine, and the header line tells you how much they cut:
`algo=MCTS+VO-TREE, scene=intention: 20 of 260 runs`.

Each takes a **comma-separated list** — `--algo MCTS,VO-TREE`, `--label B3,B4` — matched
case-insensitively. Comma-separated rather than repeated words, because `--algo MCTS VO-TREE plot`
would let argparse swallow `plot` as a third algorithm and never reach the subcommand. An entry
matching nothing gets a warning naming what *is* available, so a typo in one item of a list
doesn't quietly return fewer runs and look like a result.

`--label` matters more than it looks: without it, the sources are `debug/` **plus every
snapshot**, so `plot --anim` re-renders B1 as well as B3. `--label B3` restricts it to one, and
`--label ""` to the live folder. Do not use `--dir debug_archive/B3` for this — that discovers
those runs with no label, files them at the top level of `debug_plots/` instead of under `B3/`,
and finds them a second time through the archive scan.

**Recreating everything after deleting `debug_plots/`** — the run argument is optional, and the
sources are `debug/` plus every snapshot, so one command rebuilds the lot:

```bash
python3 summarize_debug.py plot            # ~15 s for a 180-run campaign
python3 summarize_debug.py plot --anim     # ~3 min on 8 cores, animations included
```

Timings are measured. Stills are a fraction of a second each. Animations cost about 0.026 s per
simulation step per core — 4 s for a 160-step run — and `--jobs` (default: 8, or your core count)
divides a campaign across cores: 60 runs take 54 s rather than the 8½ minutes the same work used
to take. `--limit N` caps it, `--stride 2` halves the frames, and Ctrl-C is safe — every still is
written before the first animation starts.

If you want them smaller, `--format .mp4` is about 25 % faster to write and five times smaller
(76 kB against 373 kB for a 158-frame run), at the cost of needing a player rather than any image
viewer.

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

**`--anim`** draws the same picture as `plot_frame2` in `MCTS_VO/experiment_utils.py` — the
renderer `debug_utils.py` uses at the end of a run — but builds its artists once and then updates
their data, where `plot_frame2` clears the axes and rebuilds everything each frame. That was
43 ms per frame and is now 26 ms; the rest is GIF encoding. The output is **pixel-identical** —
verified frame by frame on raw RGB buffers over 1540 frames of eight runs, covering both scenes,
all three algorithms, obstacle counts from 2 to 8, and archives with and without `obs_`.
`--anim` *adds*
animations rather than replacing the stills, and prints per-run progress, since a silent process
for minutes is hard to tell from a hung one.

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

**Discounted return is not comparable across a change of discount.** It went from 0.9 per step to
0.81 per second (0.979 per step), and each run's return was computed with whatever was active at
the time, so comparing discounted returns across that boundary is meaningless. Undiscounted return
is unaffected. The script prints a warning when a summary spans groups with different discounts —
though it can only do that for runs carrying a `gammaPerSecond` column; older ones look identical
to it, so read `undiscountedReturn` when in doubt.

**`steps` and `totalS` are not comparable across a change in `ts`.** A step is 0.1 s in the default
configuration and 0.02 s at 25 Hz, so the step count changes even when nothing about the behaviour
does. Another warning covers this.

**Nothing is comparable across a change in `obsSpeedScale`.** This is the harshest of the three,
because it is not one column: at `--obs-speed-scale 1.5` the obstacles moved half again as fast, so
the runs did not face the same scene and every number above differs for reasons that have nothing
to do with the planner. The script warns when a summary spans more than one scale. Runs recorded
before the column existed are read as 1.0, since the builds then could not honour anything else.

Related, and not something the script can warn about: runs from before the obstacle speeds were
normalised were recorded with the sinusoidal obstacles moving at 0.5078 m/s rather than 0.1, and
they carry no `obsSpeedScale` column to distinguish them. To compare against those, run at
`--obs-speed-scale 5.099`, which reproduces the old motion exactly.

**Timing rows marked `*` were reconstructed, not measured.** Runs made before the `step_stats`
instrumentation have no per-phase record, but the split is still exactly recoverable, because the
loop stored `times[i] = ts − t_sense − 0.005`. So `t_sense` and `t_plan` are real numbers, not
guesses — but `cycleMed` for those rows is the configured period rather than an observed one, and
`sims/step` comes from a separate file. Runs that do have `step_stats_*.pkl` are read from it
directly, and carry no asterisk. If a run has no `ts` column the script assumes 0.1 s; override
with `--legacy-ts`.

**Blank cells are usually correct.** `VO-PLANNER` has no tree, so depth and `sims/step` are empty
for it. A dash means "this run never recorded that", not "zero".

**A group is one configuration, not one experiment.** Runs are grouped by
`(snapshot, scene, algorithm, filename suffix)`. Two batches run at different settings but with the
same filenames land in the same group and get averaged together — which is exactly why the
snapshot command exists.

## Where things live

| | |
|---|---|
| `summarize_debug.py` | the script |
| `SUMMARIZE_README.md` | this file |
| `debug/` | live results — where each run writes, and what gets overwritten |
| `debug_archive/<label>/` | snapshots: data + `plots/` |
| `debug_plots/<algo>/<env>/<outcome>/` | output of `plot` |
| `analysis/summary.csv` | one row per group, rewritten every run |
| `analysis/runs.csv` | one row per run, rewritten every run |
| `~/.local/share/Trash/files/debug/` | earlier experiments: RADIUS_SCALE sweep, 5/10/20/25 Hz sweep |

`--dir` reads any of these in place. Nothing is ever moved or deleted; `snapshot` copies.
