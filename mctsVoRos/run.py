import argparse
import subprocess


NUM_EXP = 30

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', default='VO-TREE', type=str,
                    choices=['MCTS', 'VO-TREE', 'VO-PLANNER'])
parser.add_argument('--trajectories', default='sinusoidal', type=str,
                    choices=['sinusoidal', 'intention'])
parser.add_argument('--num_exp', default=NUM_EXP, type=int)
args = parser.parse_args()

for i in range(args.num_exp):
    print(f"Running experiment {i}")
    subprocess.run([
        "python3", "loopHandler_copy.py",
        "--exp_num", str(i),
        "--algorithm", args.algorithm,
        "--trajectories", args.trajectories,
    ])
    print(f"Experiment {i} finished")