import subprocess


NUM_EXP = 30

for i in range(NUM_EXP):
    print(f"Running experiment {i}")
    subprocess.run(["python3", "loopHandler_copy.py", "--exp_num", str(i), "--algorithm", "VO-PLANNER"])
    print(f"Experiment {i} finished")