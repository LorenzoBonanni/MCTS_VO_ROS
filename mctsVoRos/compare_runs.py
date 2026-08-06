import argparse
import pickle
import numpy as np

def load_run_data(out_dir, algo, exp_num, suffix):
    """Load all relevant pickled data from one run."""
    base = f"{out_dir}/" if not suffix.endswith("/") else out_dir
    # File names are like: trj_VO-TREE_0_run1.pkl
    tag = f"{algo}_{exp_num}{suffix}"
    trj_file = f"{base}trj_{tag}.pkl"
    acts_file = f"{base}acts_{tag}.pkl"
    obs_pred_file = f"{base}obsPred_{tag}.pkl"
    step_stats_file = f"{base}step_stats_{tag}.pkl"
    sim_num_file = f"{base}sim_num_{tag}.pkl"

    with open(trj_file, 'rb') as f:
        traj = pickle.load(f)
    with open(acts_file, 'rb') as f:
        acts = pickle.load(f)
    with open(obs_pred_file, 'rb') as f:
        obs_pred = pickle.load(f)
    with open(step_stats_file, 'rb') as f:
        step_stats = pickle.load(f)
    try:
        with open(sim_num_file, 'rb') as f:
            sim_num = pickle.load(f)
    except FileNotFoundError:
        sim_num = [np.nan] * len(traj)
    return traj, acts, obs_pred, step_stats, sim_num

def compare_runs(out_dir, algo, exp_num, suffix1, suffix2):
    traj1, acts1, obs_pred1, step_stats1, sim_num1 = load_run_data(out_dir, algo, exp_num, suffix1)
    traj2, acts2, obs_pred2, step_stats2, sim_num2 = load_run_data(out_dir, algo, exp_num, suffix2)

    min_len = min(len(traj1), len(traj2))
    diverged = False

    print(f"Comparing {algo} exp {exp_num}: '{suffix1}' vs '{suffix2}'")
    print(f"Trajectory lengths: {len(traj1)} vs {len(traj2)}")
    print("-" * 60)

    for i in range(min_len):
        # 1. Compare trajectory (robot state: x, y, yaw, velocity)
        if not np.allclose(traj1[i], traj2[i], rtol=1e-9, atol=1e-9):
            print(f"TRAJECTORY diverges at step {i}:")
            print(f"  Run1: {traj1[i]}")
            print(f"  Run2: {traj2[i]}")
            diverged = True
            break

        # 2. Compare actions (including 'None' for last step)
        act1 = acts1[i] if i < len(acts1) else None
        act2 = acts2[i] if i < len(acts2) else None
        if act1 is not None and act2 is not None:
            if not np.allclose(act1, act2, rtol=1e-9, atol=1e-9):
                print(f"ACTION diverges at step {i}:")
                print(f"  Run1: {act1}")
                print(f"  Run2: {act2}")
                diverged = True
                break
        elif act1 is None or act2 is None:
            if act1 != act2:
                print(f"ACTION presence diverges at step {i}: {type(act1)} vs {type(act2)}")
                diverged = True
                break

        # 3. Compare number of simulations (the source of divergence)
        n1 = sim_num1[i] if i < len(sim_num1) else np.nan
        n2 = sim_num2[i] if i < len(sim_num2) else np.nan
        if not np.isclose(n1, n2, rtol=0, atol=0) and not (np.isnan(n1) and np.isnan(n2)):
            print(f"SIM_COUNT differs at step {i}: Run1={n1}, Run2={n2}")
            # Not breaking here, but it's often the first sign of divergence

        # 4. Compare obstacle predictions (optional)
        obs1 = obs_pred1[i] if i < len(obs_pred1) else None
        obs2 = obs_pred2[i] if i < len(obs_pred2) else None
        if obs1 is not None and obs2 is not None:
            # Compare positions and radii
            if len(obs1[0]) != len(obs2[0]) or not np.allclose(obs1[0], obs2[0], rtol=1e-9, atol=1e-9) or \
               len(obs1[1]) != len(obs2[1]) or not np.allclose(obs1[1], obs2[1], rtol=1e-9, atol=1e-9):
                print(f"OBSTACLES differ at step {i}")
                diverged = True
                break

    if not diverged and len(traj1) != len(traj2):
        print(f"Trajectories identical for first {min_len} steps, but lengths differ: {len(traj1)} vs {len(traj2)}")
    elif not diverged:
        print("Runs are identical across all compared data (trajectory, actions, obstacles, sim counts).")
    else:
        print("First divergence shown above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two reproducibility runs.")
    parser.add_argument("out_dir", help="Output directory, e.g. debug/sinusoidal")
    parser.add_argument("algo", help="Algorithm name, e.g. VO-TREE")
    parser.add_argument("exp_num", type=int, help="Experiment number")
    parser.add_argument("suffix1", help="Suffix of first run (e.g. _run1)")
    parser.add_argument("suffix2", help="Suffix of second run (e.g. _run2)")
    args = parser.parse_args()

    compare_runs(args.out_dir, args.algo, args.exp_num, args.suffix1, args.suffix2)