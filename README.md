# Safe Monte Carlo Planning for Mobile Robots in Dynamic Environments

This repository contains the implementation for the paper "Safe Monte Carlo Planning for Mobile Robots in Dynamic Environments". The project was tested on:
- Ubuntu 20.04
- Python 3.8.10
- ROS2 Foxy
- Unity

Those are the versions the code needs, not the versions your machine needs: the
supplied Docker image provides all of them, so any Linux host with Docker can run
the experiments. See [Installation](#installation).

## Repository Structure
- `env_build/`: Contains the compiled Unity environments, one pair per variant.
  `sin_env*` is the sinusoidal obstacle trajectories environment, `int_env*` the
  intention-based one. Select with `--env-build`:
  - `*_env_fixed/`: current builds — frame-rate-independent obstacle motion,
    peak speed normalised to 0.1 m/s, sensors at 50 Hz (the default)
  - `*_env/`, `*_env_50hz/`: pre-fix builds, kept so earlier results stay reproducible
- `mctsVoRos/`: Contains the Python implementation of the algorithms and experiment runner
  - `summarize_debug.py`: reads the results and prints them as tables; see
    [`SUMMARIZE_README.md`](mctsVoRos/SUMMARIZE_README.md)
- `mcts_vo_Turtlebot3UnityROS2/`: Unity project implementing the simulation environment

## Installation

Two ways in. **Docker is the recommended one** — it needs neither Ubuntu 20.04 nor
a system ROS 2, so it works on any Linux with Docker installed, and it is the only
route that does not depend on the host having the exact versions above. Install
natively only if you want to develop against a system ROS 2 you already run.

### Option 1: Docker (recommended)

```bash
git clone --recurse-submodules https://github.com/LorenzoBonanni/MCTS_VO_ROS.git
cd MCTS_VO_ROS
docker/run.sh --build
```

That is the whole setup. `--recurse-submodules` fetches the planner
(`mctsVoRos/MCTS_VO`) at the right commit, the Unity environments are already in
the repo so nothing is compiled, and `--build` produces the image once.

The image holds only ROS 2 Foxy, Python 3.8 and the pinned packages. **The source
is bind-mounted, not baked in**, so editing code or switching between the tagged
fixes needs no rebuild — rebuild only after changing `docker/Dockerfile` or
`mctsVoRos/requirements.txt`. Run anything through the wrapper:

```bash
docker/run.sh python3 loopHandler_copy.py --algorithm VO-TREE --trajectories intention
docker/run.sh ./run_all_experiments.sh -n 30 --skip-setup
docker/run.sh python3 summarize_debug.py
docker/run.sh                                    # interactive shell
```

The working directory inside the container is `mctsVoRos/`, so every command below
works unchanged with `docker/run.sh` in front of it. `--skip-setup` is required for
the campaign script: without it the script sources ROS, runs `colcon build` and
activates `venv/`, none of which exist in the container. Results land on the host,
owned by you, in `mctsVoRos/debug/`.

If `docker` reports a permission error on `/var/run/docker.sock`, your login session
predates your `docker` group membership — log out and back in, or prefix commands
with `sg docker -c "..."`.

See [`docker/README.md`](docker/README.md) for the container details: building
without the wrapper, Podman and HPC clusters, render modes, and parallel runs.

### Option 2: Native install

Needs Ubuntu 20.04, Python 3.8 and ROS 2 Foxy already installed.

1. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

2. Clone this repository into your colcon workspace:
   ```bash
   git clone --recurse-submodules https://github.com/LorenzoBonanni/MCTS_VO_ROS.git ~/colcon_ws/src/MCTS_VO_ROS
   cd ~/colcon_ws/src/MCTS_VO_ROS
   ```
   `--recurse-submodules` populates `mctsVoRos/MCTS_VO` at the commit this
   revision expects. If you already cloned without it, run
   `git submodule update --init` rather than cloning the planner by hand.

3. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv --system-site-packages venv/
   source venv/bin/activate
   ```
   `--system-site-packages` is required so that the ROS 2 Python packages
   installed system-wide (`rclpy`, `tf_transformations`, ...) remain importable
   from inside the environment.

4. Install required Python packages:
   ```bash
   pip install -r mctsVoRos/requirements.txt
   ```

5. Build the project:
   ```bash
   colcon build
   source install/setup.bash
   ```

## Running Experiments

### Important Pre-Run Requirements

Every command below is run from the `mctsVoRos/` directory:

```bash
cd ~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos
```

With Docker that is already the working directory inside the container, so put
`docker/run.sh` in front of each command and run it from the repository root
instead. The output directories are created on demand; nothing has to exist
beforehand.

### Reproducing Paper Experiments
**Note: All commands should be run from the `~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos` directory.**

```bash
python3 run.py
```

#### Configuration Options:
1. **Obstacle Trajectories** (default: sinusoidal):
   Pass the `--trajectories` argument, no source edit is needed:
   - `sinusoidal`: sinusoidal obstacle trajectories (`env_build/sin_env`, default)
   - `intention`: intention-based obstacle trajectories (`env_build/int_env`)

   The results of a run are written to `debug/<trajectories>/`, so the two
   domains never overwrite each other (e.g. `debug/sinusoidal`, `debug/intention`).
   Animations (trajectory GIF and rollout MP4) go to `debug/<trajectories>/animations/`.

2. **Algorithm Selection** (default: MCTS-VO):
   Pass the `--algorithm` argument:
   - `VO-TREE`: MCTS-VO (default)
   - `MCTS`: Standard MCTS
   - `VO-PLANNER`: VO-Planner

Both arguments are accepted by `run.py` and by `loopHandler_copy.py`:
```bash
python3 run.py --algorithm VO-TREE --trajectories intention
```

### Search Depth Metrics
For the tree-based planners (`MCTS` and `VO-TREE`) each run also reports how deep
the search went. Three quantities are recorded at every planning step:

| Metric | Meaning |
| --- | --- |
| `max_tree_depth` | depth of the deepest node of the search tree (root = 0) |
| `max_rollout_depth` | length in steps of the longest rollout |
| `max_total_depth` | deepest state reached overall (depth at which a rollout started + its length) |

The per-step values are saved to `debug/<trajectories>/depths_<ALGORITHM>_<N>.pkl`,
and their run-level aggregates appear in `data_<ALGORITHM>_<N>.csv` as
`maxTreeDepth`, `meanTreeDepth`, `maxRolloutDepth`, `meanRolloutDepth`,
`maxTotalDepth` and `meanTotalDepth`. `VO-PLANNER` builds no tree, so these
columns are `NaN` for it.

The counters are updated only when a node is created and at the end of a rollout,
never on the per-visit path of `simulate()`, so they do not slow down the planner.

### Running the Full Experimental Campaign
To reproduce every configuration of the paper (3 algorithms x 30 runs x 2 domains
= 180 runs) in a single command:
```bash
./run_all_experiments.sh
```
The script is self-contained and can be launched from any directory: it sources
`/opt/ros/$ROS_DISTRO/setup.bash`, runs `colcon build` in the workspace root,
sources `install/setup.bash`, activates `venv/` and then runs the experiments.
Nothing has to be sourced or activated beforehand.

Results of each run go to `debug/<trajectories>/` (animations in
`debug/<trajectories>/animations/`) and the console output to
`logs/<trajectories>/`. Completed runs are detected and skipped, so the campaign
can be resumed after an interruption; use `--force` to re-run them.

Useful options (see `./run_all_experiments.sh --help` for the full list):

| Option | Effect |
| --- | --- |
| `-n, --num-exp N` | runs per configuration (default: 30) |
| `-a, --algorithms "A B"` | subset of algorithms |
| `-t, --trajectories "X Y"` | subset of domains |
| `-f, --force` | re-run configurations that already have results |
| `-x, --extra "ARGS"` | extra arguments passed straight to `loopHandler_copy.py` |
| `--skip-build` | source and activate, but skip `colcon build` |
| `--skip-setup` | run in the current shell, without sourcing/building/activating |
| `--ros-distro NAME` | ROS 2 distribution (default: `$ROS_DISTRO`, else `foxy`) |

`-x` is how per-run options reach the planner from a campaign, for example
`-x "--obs-speed-scale 1.5"` to run the whole campaign against faster obstacles.

The virtual environment must be created with `--system-site-packages`, otherwise
the ROS 2 Python packages (`rclpy`, `tf_transformations`, ...) are not visible
from inside it. The script checks this before starting and aborts with an
explanatory message if the environment is incomplete.

### Running Single Experiments
```bash
python3 loopHandler_copy.py --exp_num <EXPERIMENT_NUMBER> --algorithm <ALGORITHM> --trajectories <TRAJECTORIES>
```
Example:
```bash
python3 loopHandler_copy.py --exp_num 1 --algorithm VO-TREE --trajectories sinusoidal
```

Run `python3 loopHandler_copy.py --help` for the full list; each option carries the
reasoning for its default. The ones that change what is being measured, rather
than how much of it:

| Option | Effect |
| --- | --- |
| `--obs-speed-scale K` | multiplies the speed of every obstacle in either scene (default 1) |
| `--env-build {fixed,orig,50hz}` | which simulator build to launch |
| `--env-render {window,headless}` | whether the simulator draws a window |

### Obstacle speed

Both scenes are normalised so that every obstacle peaks at exactly `0.1 * K` m/s,
and `--max-obs-vel` — the speed the velocity obstacles are sized for — is derived
from the same number, so the planner and the simulator cannot disagree:

```bash
python3 loopHandler_copy.py --algorithm VO-TREE --trajectories sinusoidal --obs-speed-scale 1.5
```

`K` scales the *step period*, not the velocity, so the obstacles walk exactly the
same paths on a faster clock: the geometry, the waypoints and the random draws are
unchanged, and only the pace differs. The robot itself tops out at 0.3 m/s, so
past `K = 3` it has no speed advantage left to avoid with.

Two things to know before comparing across it. Runs at different `K` did not face
the same scene, so their outcomes are not comparable — `summarize_debug.py` warns
when a summary mixes them. And runs recorded before the speeds were normalised had
the sinusoidal obstacles moving at 0.5078 m/s rather than 0.1; use
`--obs-speed-scale 5.099` to reproduce that exactly.

The scale reaches the simulator as `-obsSpeedScale` and is recorded in every
result CSV as `obsSpeedScale`, beside `maxObsVel`. Only builds from 2026-08-04
onwards honour it — older ones ignore the argument.

## Reading the results

`mctsVoRos/summarize_debug.py` reads the run artefacts in `debug/` and prints five
tables: outcomes, returns, timing, search depth and motion smoothness. It also
snapshots a `debug/` folder under a label, so re-running after each fix does not
silently overwrite the previous results.

```bash
cd mctsVoRos
source ../venv/bin/activate
python3 summarize_debug.py
```

**See [`mctsVoRos/SUMMARIZE_README.md`](mctsVoRos/SUMMARIZE_README.md) for the full
guide** — what every column means and which direction is better, the snapshot
workflow for stepping through the fixes, the plotting and animation commands, and
the comparisons that will mislead you if you make them.

## Final Directory Structure

After completing the installation steps, your directory structure should look like
this. `build/`, `install/` and `log/` are produced by `colcon build` and so appear
only with the native install; the Docker route never creates them.

```
~/colcon_ws/src/MCTS_VO_ROS/
├── docker/
│   ├── Dockerfile
│   ├── run.sh
│   └── README.md
├── env_build/
│   ├── sin_env/            # pre-fix builds
│   ├── int_env/
│   ├── sin_env_50hz/
│   ├── int_env_50hz/
│   ├── sin_env_fixed/      # current builds, used by default
│   └── int_env_fixed/
├── mctsVoRos/
│   ├── MCTS_VO/
│   │   ├── bettergym/
│   │   ├── environment_creator.py
│   │   ├── experiment_utils.py
│   │   ├── __init__.py
│   │   └── mcts_utils.py
│   ├── debug/
│   │   ├── sinusoidal/
│   │   │   └── animations/
│   │   └── intention/
│   │       └── animations/
│   ├── logs/
│   ├── debug_utils.py
│   ├── estimate_obs.py
│   ├── __init__.py
│   ├── loopHandler_copy.py
│   ├── requirements.txt
│   ├── run.py
│   ├── run_all_experiments.sh
│   ├── summarize_debug.py
│   └── SUMMARIZE_README.md
├── build/
├── install/
├── log/
├── mcts_vo_Turtlebot3UnityROS2/
├── package.xml
├── README.md
├── resource/
├── setup.cfg
├── setup.py
├── test/
└── venv/
```

Always make sure to run the experiments from the `~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos` directory to ensure correct path references.