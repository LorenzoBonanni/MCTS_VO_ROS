# Safe Monte Carlo Planning for Mobile Robots in Dynamic Environments

This repository contains the implementation for the paper "Safe Monte Carlo Planning for Mobile Robots in Dynamic Environments". The project was tested on:
- Ubuntu 20.04
- Python 3.8.10
- ROS2 Foxy
- Unity

## Repository Structure
- `env_build/`: Contains the compiled Unity environments
  - `sin_env/`: Sinusoidal obstacle trajectories environment
  - `int_env/`: Intention-based obstacle trajectories environment
- `mctsVoRos/`: Contains the Python implementation of the algorithms and experiment runner
- `mcts_vo_Turtlebot3UnityROS2/`: Unity project implementing the simulation environment

## Installation

1. Install system dependencies:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

2. Clone this repository into your colcon workspace:
   ```bash
   git clone https://github.com/LorenzoBonanni/MCTS_VO_ROS.git ~/colcon_ws/src/MCTS_VO_ROS
   ```

3. Navigate to the project directory:
   ```bash
   cd ~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos
   ```

4. Remove the existing MCTS_VO directory and clone the ROS branch:
   ```bash
   rm -rf MCTS_VO
   git clone -b ros https://github.com/Isla-lab/MCTS_VO.git
   ```

5. Return to the main project directory:
   ```bash
   cd ..
   ```

6. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv --system-site-packages venv/
   source venv/bin/activate
   ```
   `--system-site-packages` is required so that the ROS 2 Python packages
   installed system-wide (`rclpy`, `tf_transformations`, ...) remain importable
   from inside the environment.

7. Install required Python packages:
   ```bash
   pip install -r mctsVoRos/requirements.txt
   ```

8. Build the project:
   ```bash
   colcon build
   source install/setup.bash
   ```

## Running Experiments

### Important Pre-Run Requirements
1. Navigate to the mctsVoRos directory:
   ```bash
   cd ~/colcon_ws/src/MCTS_VO_ROS/mctsVoRos
   ```

2. Create debug directory:
   ```bash
   mkdir debug
   ```

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
| `--skip-build` | source and activate, but skip `colcon build` |
| `--skip-setup` | run in the current shell, without sourcing/building/activating |
| `--ros-distro NAME` | ROS 2 distribution (default: `$ROS_DISTRO`, else `foxy`) |

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

## Final Directory Structure

After completing the installation steps, your directory structure should look like this:

```
~/colcon_ws/src/MCTS_VO_ROS/
├── env_build/
│   ├── sin_env/
│   └── int_env/
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
│   └── run_all_experiments.sh
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