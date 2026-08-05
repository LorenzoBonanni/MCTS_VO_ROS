# Parameter sweeps: `docker/sweep.sh`

Sweeps `RADIUS_SCALE`, the discount and the UCB exploration constant, several
containers at a time, **one directory per parameter set**.

This branch is `b9-sweep` — tag `B9`, plus `--no-plots`, four extra CSV columns
and this script. Nothing in the planner, the environment, the reward or the VO
geometry differs from B9, and no default changes.

## Setup

```bash
git fetch --all --tags --force
git checkout b9-sweep
git submodule update --init      # not optional, see below
docker/run.sh --build            # only if the image is not built yet
```

`git submodule update --init` is not optional: `mctsVoRos/MCTS_VO` is a
submodule, a checkout does not populate it, and without it every run dies on
`No module named 'MCTS_VO.experiment_utils'`. The script refuses to start
without it.

## Running

```bash
docker/sweep.sh --dry-run     # print the plan, run nothing
docker/sweep.sh               # ask, then run
docker/sweep.sh -y            # do not ask
```

Defaults: 3 containers, 20 runs per configuration, `VO-TREE`, both scenes.

| | values | B9 default |
|---|---|---|
| `--radius-scale` | 1.4 1.8 2.2 2.6 | 1.8 |
| `--gamma` | 0.65 0.81 0.90 0.95 | 0.81 |
| `--exploration-c` | 0.5 1.0 2.0 5.0 | 1.0 |

**One axis at a time**, not the product: vary one, hold the other two at the B9
default. Ten configurations rather than sixty-four, with the baseline shared by
the three axes and run once. `--grid` takes the product instead — 64
configurations, 2560 runs; do not reach for it by accident.

Ten configurations × 1 algorithm × 2 scenes × 20 runs = 400 runs, roughly
3.5–4 h on three containers. The script prints its own estimate first.

Other options: `-n` runs each, `-j` containers, `-a` algorithms, `-t` scenes,
`-o` output directory, `--extra "ARGS"` passed to every run. To halve it, use
one scene: `docker/sweep.sh -t intention`.

## Stopping it

**Ctrl-C.** The containers are labelled, so the script stops exactly its own on
the way out — another sweep beside it, or an unrelated `docker/run.sh`, is left
alone.

Nothing is lost. Every finished run is already written to its own directory on
the host. Rerun with the same `-o` and finished runs are skipped; only the
interrupted one is redone.

If the driver was killed outright (`kill -9`, a closed terminal) the trap never
ran and containers may survive:

```bash
docker ps --filter label=mctsvo-sweep
docker stop $(docker ps -q --filter label=mctsvo-sweep)
```

## Each parameter set gets its own directory

```
sweeps/<date>-<time>/
    configs.tsv              what was run
    sweep.log                which configuration started and finished when
    summary_all.csv          EVERY CONFIGURATION IN ONE TABLE   <- read this
    runs_all.csv             every individual run
    rs1.8_g0.81_c1.0/        <- one directory per parameter set
        config.env               the exact arguments, machine-readable
        debug/<scene>/           the run data: CSVs and pickles
        logs/<scene>/            one console log per run
        run.log                  the campaign's own output
        summary.csv  runs.csv    this configuration alone
```

The name carries all three values, so a directory says what produced it. The
first four columns of `summary_all.csv` and `runs_all.csv` are `config`,
`radius_scale`, `gamma_per_second`, `exploration_c`.

The separate directories are not tidiness, they are what makes running three at
once **correct**. `loopHandler_copy.py` writes to a hardcoded `debug/` and names
its files by algorithm and run number only — never by the parameters — so three
containers sharing one checkout would silently overwrite each other. Each
container gets its own `debug/` and `logs/` bind-mounted over the repository's.
The checkout itself, Unity builds included, stays shared.

Everything else that could collide is already isolated by the container: DDS
discovery is per network namespace, so Unity and the planner only ever find
their own partner; `run_all_experiments.sh`'s Unity cleanup `pkill` is per PID
namespace, so it cannot kill another configuration's environment; and
`NUMBA_CACHE_DIR` points inside the image rather than into the mount.

`sweeps/` is gitignored.

## Before reading the results

**1. Running three at once makes every run worse, equally.** The planner works
to a *time* budget, so a container with a third of the machine gets through
fewer simulations per step and plans worse. Each container is pinned to its own
fixed cores so the handicap is the same for every configuration — the
configurations are comparable **with each other**, and **not** with a run made
on an idle machine. Check `sims` in `summary_all.csv` is roughly constant across
configurations; if it is not, the differences may be CPU rather than parameters.
Re-run a winner on its own before believing it.

**2. Discounted return is not comparable across the gamma axis.** Only there —
the radius and exploration axes are fine. On the gamma axis read
`undiscReturn`, `goalPct`, `volCollPct` and the smoothness columns. The summary
now warns about this, which plain B9 could not: the warning keys off a
`gammaPerSecond` column that nothing was writing, which is what the new CSV
columns are for.

**3. n = 20 is about 5 percentage points per run.** Runs are not reproducible —
`exp_num` seeds the planner, not the simulator — so treat a 5–10 point gap as
noise unless a test says otherwise.

**4. Every run is headless and has no animation.** The data is all there; draw
one afterwards with

```bash
docker/run.sh python3 summarize_debug.py plot --anim \
    --dir /ws/src/MCTS_VO_ROS/sweeps/<date>-<time>/rs1.8_g0.81_c1.0/debug
```

## Looking at the results

`summary_all.csv` opens in a spreadsheet, one configuration per row per group.
Worth sorting on: `goalPct`, `volCollPct`, `obsCollPct`, `timeoutPct`,
`undiscReturn`, `m_vsm`, `m_hsm`, `sims`.

One configuration in the terminal:

```bash
docker/run.sh python3 summarize_debug.py --no-archive --runs \
    --dir /ws/src/MCTS_VO_ROS/sweeps/<date>-<time>/rs1.8_g0.81_c1.0/debug
```

## If something looks wrong

| | |
|---|---|
| `The MCTS_VO submodule is not populated` | `git submodule update --init` |
| `Image mctsvo:foxy not found` | `docker/run.sh --build` |
| `NO RUN PRODUCED ANY DATA` / `INCOMPLETE: n of m` | the traceback is at the **end** of `<config>/logs/<scene>/<algo>_0.log` |
| `includes invalid characters for a local volume name` | fixed 2026-08-05; `git pull` |
| Output owned by root | the image was built by another user: `docker/run.sh --build` |
| Everything slow, `sims` low | something else is using the machine; run fewer containers with `-j 2` |
| A Unity process survives | `docker ps --filter label=mctsvo-sweep`, then `docker stop ...` |
| `permission denied ... docker.sock` | the login session predates your docker group membership; log out and back in, or `sg docker -c "docker/sweep.sh ..."` |

Container detail: `docker/README.md`. Reading the results:
`mctsVoRos/SUMMARIZE_README.md`.
