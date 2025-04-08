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
   git clone https://github.com/LorenzoBonanni/MCTS_VO_ROS.git colcon_ws/src
   ```

3. Navigate to the project directory:
   ```bash
   cd colcon_ws/src/MCTS_VO_ROS/mctsVoRos
   ```

4. Remove the existing MCTS_VO directory and clone the ROS branch:
   ```bash
   rm -rf MCTS_VO
   git clone -b ros https://github.com/Isla-lab/MCTS_VO.git
   ```

5. Return to the main project directory:
   ```bash
   cd ../..
   ```

6. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv/
   source venv/bin/activate
   ```

7. Install required Python packages:
   ```bash
   pip install -r python_code/requirements.txt
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
   cd colcon_ws/src/MCTS_VO_ROS/mctsVoRos
   ```

2. Create debug directory:
   ```bash
   mkdir debug
   ```

### Reproducing Paper Experiments
```bash
python3 run.py
```

#### Configuration Options:
1. **Obstacle Trajectories** (default: sinusoidal):
   Modify `loopHandler_copy.py`:
   ```python
   # For sinusoidal obstacles (default)
   process = subprocess.Popen(["../env_build/sin_env/env.x86_64"], preexec_fn=os.setpgrp)
   
   # For intention-based obstacles
   # process = subprocess.Popen(["../env_build/int_env/env.x86_64"], preexec_fn=os.setpgrp)
   ```

2. **Algorithm Selection** (default: MCTS-VO):
   Modify the `--algorithm` argument in `run.py`:
   - `VO-TREE`: MCTS-VO (default)
   - `MCTS`: Standard MCTS
   - `VO-PLANNER`: VO-Planner

### Running Single Experiments
```bash
python3 loopHandler_copy.py --exp_num <EXPERIMENT_NUMBER> --algorithm <ALGORITHM>
```
Example:
```bash
python3 loopHandler_copy.py --exp_num 1 --algorithm VO-TREE
```
