"""Old three-function VO pipeline vs the fused numba one, on real recorded frames."""
import glob, os, pickle, sys
import numpy as np

sys.path.insert(0, '/home/lorenzobonanni/Desktop/MCTS_VO_ROS/mctsVoRos')

from MCTS_VO.bettergym.agents.utils.vo import (
    get_radii, get_unsafe_angles, compute_ranges_difference)
from MCTS_VO.bettergym.agents.utils.utils import get_robot_angles
from MCTS_VO.mcts_utils import get_intersections_vectorized
from MCTS_VO.bettergym.compiled_utils import vo_forbidden_ranges

DT, RR, VMAX, TM, MAC = 0.1, 0.15, 0.22, 0.1, 0.284


def norm(ranges):
    """Canonical form so two range sets can be compared regardless of order."""
    a = np.asarray(ranges, dtype=float).reshape(-1, 2)
    if len(a) == 0:
        return a
    return a[np.lexsort((a[:, 1], a[:, 0]))].round(9)


def old_path(x, ox, r0, r1, ra):
    inter, _, _ = get_intersections_vectorized(x, ox, r0, r1)
    forb = get_unsafe_angles(inter, ra, x)
    safe = compute_ranges_difference(ra, forb)
    return norm(forb), norm(safe)


def new_path(x, ox, r0, r1, ra):
    forb = vo_forbidden_ranges(x, ox, r0, r1)
    safe = compute_ranges_difference(ra, forb)
    return norm(forb), norm(safe)


frames = same_forb = same_safe = 0
diff_forb_examples = []
diff_safe_examples = []

for scene in ('intention_complex', 'sinusoidal_complex'):
    for f in sorted(glob.glob(f'debug/{scene}/trj_*.pkl')):
        suf = os.path.basename(f)[4:-4]
        try:
            trj = pickle.load(open(f, 'rb'))
            obs = pickle.load(open(f.replace('trj_', 'obs_'), 'rb'))
        except Exception:
            continue
        for i in range(min(len(trj), len(obs))):
            x = np.asarray(trj[i], dtype=np.float64)
            ox, orad = obs[i]
            if len(ox) == 0:
                continue
            ox = np.ascontiguousarray(ox, dtype=np.float64)
            r1, r0 = get_radii(ox, np.asarray(orad, float), DT, RR, VMAX,
                               think_margin=TM)
            ra = np.asarray(get_robot_angles(x, MAC), dtype=np.float64)

            fo, so = old_path(x, ox, r0, r1, ra)
            fn, sn = new_path(x, ox, r0, r1, ra)

            frames += 1
            ok_f = fo.shape == fn.shape and np.allclose(fo, fn, atol=1e-9)
            ok_s = so.shape == sn.shape and np.allclose(so, sn, atol=1e-9)
            same_forb += ok_f
            same_safe += ok_s
            if not ok_f and len(diff_forb_examples) < 4:
                diff_forb_examples.append((scene, suf, i, fo, fn))
            if not ok_s and len(diff_safe_examples) < 4:
                diff_safe_examples.append((scene, suf, i, so, sn))

print(f'frames compared      : {frames}')
print(f'forbidden ranges match: {same_forb}  ({100*same_forb/frames:.3f} %)')
print(f'safe ranges match     : {same_safe}  ({100*same_safe/frames:.3f} %)')

for tag, ex in (('FORBIDDEN', diff_forb_examples), ('SAFE', diff_safe_examples)):
    for scene, suf, i, a, b in ex:
        print(f'\n{tag} mismatch  {scene}/{suf} step {i}')
        print('  old:', a.tolist())
        print('  new:', b.tolist())
