# MCTS-VO — performance work and defects found

Working notes from a session that set out to speed up the plan–sense–control loop
and ended up also finding why MCTS-VO never reached the goal.

Everything below is measured on this machine (8 cores, Ubuntu, Unity 2021.3.14f1)
against the `sinusoidal` domain with `--algorithm VO-TREE` (the paper's MCTS-VO),
5 runs per configuration unless stated otherwise. **5 runs is a small sample** —
treat percentages as indicative, not final.

---

## 1. Outcome

| configuration | goal | voluntary coll. | involuntary coll. |
|---|---|---|---|
| as shipped (wrong goal, paper hyper-parameters) | 0 % | 0 % | 80 % |
| goal corrected, paper hyper-parameters | 20 % | **20 %** | 60 % |
| goal corrected + `c=1`, `γ_step=0.979` | **80 %** | 0 % | 0 % |
| the above + `max_obs_vel=0.15` (sound VO bound) | 60 % | 0 % | 0 % |

Timing, live, per planning step:

| | before | after |
|---|---|---|
| `t_sense` | 24.3 ms mean, 41.9 ms p99 | **0.16 ms** med, 0.56 ms p99 |
| simulations / step | 32 median | **480** median |
| control cycle | 200 ms | 200 ms default; 115 ms with `--plan-budget 0.010` |

---

## 2. Defects found

### 2.1 The goal coordinate was wrong  *(fixed)*

`LoopHandler.goal` was `[-3.26, -1.61]`. The `Goal` object in
`Assets/Scenes/turtlebot3_COPY.unity` sits at planner coordinates
`(-2.783, -0.720)` — **0.95 m away**. Since `GoalAndCollisionHandler.cs` is an
empty stub, "goal reached" is judged purely in Python against `self.goal`, so the
robot was steering at a point nothing in the scene marked.

The frame mapping was verified independently rather than assumed: the robot's
spawn transform `unity(1.136, 0, 0.490)` maps to `(0.490, -1.136)`, which matches
the first odometry reading exactly. So `px = unity_z, py = -unity_x` is correct.

Fixing the goal alone took success from 0 % to 20 %.

### 2.2 `gt_obs_pos` did not match the scene  *(fixed)*

None of the six hard-coded obstacle positions corresponded to anything in the
scene. Measured against them, only **0.9 %** of LIDAR detections landed within
15 cm; against the real scene geometry, **51.7 %**. This does not affect the
planner (which uses LIDAR estimates) but it made every perception diagnostic look
broken when perception was fine, and it invalidated an early `RADIUS_SCALE`
calibration.

Real static obstacles, planner frame, all radius 0.100 m:

```
(-0.399,  0.420)  (-1.542, -1.790)  (-1.539,  0.360)
(-2.640, -1.310)  (-0.317, -1.820)  (-3.020,  0.363)
```

Four `Obstacle_*_MOVING` spheres also exist — deliberately not listed, since a
fixed position would be ground truth only at t=0.

### 2.3 UCB exploration constant swamps the signal  *(exposed as a flag)*

Root-action Q-values span about **0.062** (std 0.018). The UCB bonus at `c=10`
with ~20 visits per action is about **5.3** — roughly 100× the signal. Action
selection was therefore close to random: the planner turned *toward* the goal in
only **44–48 %** of steps, and the heading it chose was no better than the one it
already had (48.6° vs 49.5° error, when 41.3° was available).

Now `--exploration-c`, default 10 (unchanged). `c≈1` is what worked.

### 2.4 Discount gives a ~1 s effective horizon  *(exposed as a flag)*

`γ = 0.9` per step at `ts=0.1` means an effective horizon of `1/(1-γ) = 10` steps
= 1 s. The robot covers 0.22 m in that time against a 3.3 m goal, so the first
action barely changes the return — which is why the Q-values are so tightly
bunched. Now `--gamma-per-second`, default `0.9**10 = 0.349` (unchanged);
`0.81` (i.e. `γ_step = 0.979`) is what worked.

### 2.5 Two obstacles move faster than VO assumes  *(exposed as a flag)*

In the sinusoidal scene:

| obstacle | script | speed |
|---|---|---|
| Obstacle_7_MOVING | `move_1.cs` | **0.10–0.15 m/s** |
| Obstacle_8_MOVING | `move_2.cs` | **0.10–0.15 m/s** |
| Obstacle_9_MOVING | `move_copy.cs` | 0.0–0.1 m/s |
| Obstacle_10_MOVING | `move_4_copy.cs` | 0.05–0.1 m/s |

`move_1.cs` applies it as a genuine speed
(`pos.z += velocity * dt * cos(angle)`), while the planner was told
`max_obs_vel = 0.1`. The VO guarantee requires that value to be a true upper
bound, so it did not strictly hold.

Quantitatively the error is small — the keep-out radius grows only
0.270 → 0.280 m (+10 mm, 3.7 %), because the travel term is minor next to
`r_obs + r_robot`. But it is the difference between the guarantee holding and
not. Now `--max-obs-vel`, defaulting to the honest **0.15**. Cost: 80 % → 60 %
success, collisions stay at 0 %.

The alternative fix is to cap `move_1`/`move_2` at 0.1 m/s in Unity, which is
arguably what the scene intended given the other two scripts already do.

### 2.6 Episode length did not scale with `ts`  *(fixed)*

`MAX_STEPS = 350` was a step count, so halving `ts` halved the distance budget:
at `ts=0.05` an episode allowed 17.5 s of motion (3.85 m of travel) versus 35 s
at `ts=0.1`, for a 3.30 m goal. Any reduced-`dt` comparison made before this fix
was unfair. Now derived from `EPISODE_S = 35.0`.

### 2.7 Every run faces the same obstacle trajectories  *(open)*

`Random.InitState(42)` in every movement script's `Start()` seeds Unity's global
RNG, so the sequence of obstacle speeds — and therefore the path each obstacle
traces — is the same in every run.

Measured directly, with the planner taken out of the loop: the environment was
launched three times, the robot held stationary, and raw `/scan` recorded for
20 s each.

| pair | median \|Δrange\| | p90 | beams agreeing within 2 cm |
|---|---|---|---|
| A–B | 0.02 cm | 0.8 cm | 91.0 % |
| A–C | 0.00 cm | 3.4 cm | 89.0 % |
| B–C | 0.02 cm | 0.3 cm | 91.3 % |

For scale, *within* one run the scan changes 0.01 cm median / 0.2 cm p90 per
0.1 s — the difference between launches is the same order as the difference
between consecutive frames of a single launch. **The scene replays
identically.**

This matters for how the results should be read: the 30 experiments are 30
near-repeats of one scenario, not 30 samples of a randomised one, so the paper's
attribution of variance to "randomized sinewave trajectories of dynamic
obstacles" does not hold.

A caution against the obvious counter-measurement: in *full* runs the estimated
obstacle positions do differ between runs (7–14 cm). That is the closed loop
amplifying microscopic timing differences chaotically — the robot diverges, so it
observes from a different viewpoint — not the environment behaving differently.
Measuring this with the planner in the loop gives the wrong answer.

Two separate issues follow, and only the first is fixed by seeding:

- **Trajectory diversity**: `Random.InitState(42 + expNum)`, with the run index
  passed into the build, would give each run a genuinely different trajectory.
- **Reproducibility**: seeding will *not* deliver it. Chaotic divergence in the
  closed loop means `--exp_num 3` still will not repeat. That needs lockstep
  between Unity and the planner (fixed timestep, no wall-clock budget), which is
  a much larger change.

### 2.8 Sensor publish rate caps the control cycle  *(build produced, not yet validated)*

Unity published `/scan` and `/odom` at 10 Hz. `control_loop` returns early when
there is no new data, so with `t_timer = 65 ms` every other tick was skipped and
the measured cycle was exactly 130 ms. **The sensor rate, not compute, is the
binding constraint below ~100 ms.**

Rates raised to 50 Hz in both scenes and the robot prefab; rebuilt into
`env_build/sin_env_50hz/` and `env_build/int_env_50hz/`. Rule of thumb: sensor
rate ≥ 2 / (ts + think_margin).

### 2.9 Smaller items, flagged but not changed

- **Returns are not comparable to the paper.** `max_eudist` is overridden in
  `LoopHandler` to the *initial* distance (3.30 m) instead of the arena diagonal
  (14.14 m) used in `Env.__init__`, so discounted returns come out around 41
  rather than ~9.4.
- **Goal tolerance mismatch**: the loop declares success at `d <= 0.2`, the
  simulator's own terminal test is `d <= robot_radius = 0.15`.
- **`check_coll_vectorized` ignores `obs_size`**: it compares centre distance
  against `robot_radius` alone, so simulated collision checks disregard obstacle
  radius. Deliberately preserved to keep the optimisation behaviour-neutral.
- **`get_discrete_space` is order-dependent**: it floors even-indexed ranges and
  ceils odd-indexed ones, so the action set depended on `IntervalTree`'s *set*
  iteration order. Now deterministic (sorted); changed the action set in 3 of 500
  sampled states, always by which range got the extra sample.
- **`uniform_towards_goal_vo` returns a Python list** on one branch where every
  sibling returns an ndarray. Harmless today (VO-PLANNER-only path).

---

## 3. Optimisations

All verified behaviour-neutral against pristine `HEAD` where a comparison exists.

| change | file | effect |
|---|---|---|
| fused rollout into one `@njit` call | `compiled_utils.py::fused_rollout` | 200-step rollout ~1400 µs → **10.2 µs** |
| VO geometry fused | `compiled_utils.py::vo_forbidden_ranges` | part of 175 → 69.7 µs per node |
| interval subtraction replaces `IntervalTree` | `vo.py::compute_ranges_difference` | 37.6 → **5.1 µs**, 0/4000 mismatches |
| compiled discrete action set | `compiled_utils.py::discrete_actions` | bit-identical to numpy `linspace` |
| lazy unpruned action set | `env.py::get_actions_discrete_vo2` | skips 26 µs when VO prunes |
| cheaper `State.__hash__` | `env.py` | 3× (the old form expanded bytes to one int per byte) |
| debug trajectory bookkeeping behind a flag | `planner_mcts.py` | +8 % |
| horizon in seconds, not steps | `loopHandler_copy.py::HORIZON_S` | keeps horizon fixed when `ts` changes |
| segmentation + closed-form circle fit replaces HDBSCAN + RANSAC | `compiled_utils.py::segment_and_fit_circles` | 15.9 ms → **0.024 ms**, same detection count |

End to end: **54 → 548 simulations** in the same 71 ms budget at the paper's
depth 200.

Notes:

- The rollout depth reduction (200 → 60) was worth 2.5× *before* the rollout was
  compiled and only ~2 % after, so the paper's 20 s horizon was kept.
- The fused rollout is **distributionally** equivalent to the Python one, not
  bitwise: the Python version draws the ε-coin from Python's `random` and speeds
  from numba's RNG, the fused one draws both from numba's. Verified over 8 start
  states × 400 rollouts, all |z| < 1.4.
- HDBSCAN's cost is ~15 ms of *fixed* overhead, independent of point count
  (14.5 ms at 34 points, 18.2 ms at 256).

---

## 4. `RADIUS_SCALE`

Calibrated against real scans matched to real scene geometry. Tight matches
(<5 cm, n=349) give a raw fitted radius of **0.0857** against a true 0.100, i.e.
the fitter is ~15 % low, not 2× low.

| basis | scale at median | scale for 95 % cover |
|---|---|---|
| tight matches (<5 cm) | **1.17** | 2.24 |
| loose matches (<15 cm) | 1.62 | 2.19 |

The shipped value of 3 inflates a 0.100 m obstacle to ~0.19 m. Sweep at
`ts=0.1`: reversing falls from 49.7 % (scale 3.0) to 9.7 % (scale 1.5) — that is
the "bouncing off something invisible" behaviour. **2.2** is a reasonable
compromise; below ~1.5 collisions start appearing.

Caution: an earlier calibration giving 1.9/2.1 was matched against the stale
`gt_obs_pos` and is void. Perception accuracy overall: 18.5 % of detections
within 5 cm of a true obstacle, 34.2 % within 10 cm, 44.7 % within 15 cm —
inflated at the tail because 4 of 10 obstacles move and are compared against
start positions.

---

## 5. Building the Unity environment

Works headlessly. `Assets/Editor/BuildScript.cs`:

```
~/Unity/Hub/Editor/2021.3.14f1/Editor/Unity -batchmode -nographics -quit \
  -projectPath <project> -executeMethod BuildScript.BuildLinux \
  -buildScene Assets/Scenes/turtlebot3_COPY.unity \
  -buildOutput ../env_build/sin_env_50hz/env.x86_64 -logFile <log>
```

**Licensing gotcha.** Entitlement (Personal) licensing fails in batchmode with
`Error: Access token is unavailable`, even when signed into Unity Hub and even
though `--showEntitlements` reports `License Type: Assigned` including
`com.unity.editor.headless`. The Editor exits at the licence check before
reaching `-executeMethod`. Signing into the Hub does not expose the token to a
non-GUI process.

Fix — acquire a legacy licence file once:

```
~/Unity/Hub/Editor/2021.3.14f1/Editor/Data/Resources/Licensing/Client/Unity.Licensing.Client \
  --activate-ulf --username '<account>' --password '<password>'
```

This writes `~/.local/share/unity3d/Unity/Unity_lic.ulf`, after which headless
builds work with no credentials. Build settings only register the sinusoidal
scene, so the intention environment must be built by passing the other scene.

---

## 6. Diagnostic gotchas

- **Numba cache poisoning.** `MCTS_VO` uses
  `try: from MCTS_VO… except ModuleNotFoundError: from bettergym…`. Running
  anything from *inside* `MCTS_VO/` compiles under the `bettergym.*` names and
  writes those to the on-disk cache; later runs from `mctsVoRos/` then fail with
  `ModuleNotFoundError`. Clear with
  `find . -name __pycache__ -type d -exec rm -rf {} +`.
- **`test_mcts_v2.py` is stale** — it passes `s0=` to `Mcts.__init__`, which is
  not a parameter. It fails identically at `HEAD`; not a regression.
- **Timing measurements need interleaving.** Identical configurations measured
  in separate processes varied by ~20 % (43 vs 51 sims). All figures here come
  from interleaved, multi-repetition runs with medians and IQRs.
- **Collisions must be split.** `collision` (voluntary: `last_action[0] != 0`)
  versus `Obscollision` (an obstacle hit a stopped robot). Lumping them together
  produced a spurious "40 % collision" reading and hid the fact that voluntary
  collisions were 0 % throughout — i.e. VO pruning was working.

---

## 7. New command-line flags

All default to the previous behaviour, so nothing changes unless asked.

| flag | default | purpose |
|---|---|---|
| `--ts` | 0.1 | control / simulation step |
| `--plan-budget` | none | wall-clock seconds for the planner; this, not planner speed, sets cycle time |
| `--radius-scale` | 3.0 | obstacle radius inflation |
| `--max-obs-vel` | **0.15** | max obstacle speed VO is sized for (changed: was 0.1, which was not a true bound) |
| `--exploration-c` | 10.0 | UCB constant |
| `--gamma-per-second` | 0.349 | discount per second |
| `--rollout-collision` | check | `check` \| `none` |
| `--env-build` | per-domain | select a Unity build |
| `--no-plots` | off | skip animations (they take longer than the run) |
| `--collect-trajectories` | off | record simulated states for the tree animation |
| `--suffix` | "" | tag output files so sweeps do not overwrite |

---

## 8. Open

1. **`max_obs_vel = 0.15` vs capping the Unity scripts at 0.1.** Currently 0.15
   (sound guarantee, 60 % success). Capping in Unity restores 80 % and matches
   what the other two scripts already do.
2. **Per-run seeding** (§2.7) — gives each run a different trajectory; needs a
   scene change and rebuild. Note it does *not* buy reproducibility.
3. **50 Hz build not yet validated.** Built but the confirming runs have not been
   completed; expect the cycle to drop from 130 ms to ~65 ms at `ts=0.05`.
4. **`c` and `γ` are departures from the published hyper-parameters.** They are
   what takes success from 0 % to 80 %, but adopting them is a method change.
5. **Only `sinusoidal`, only 5 runs.** The `intention` domain and a full 30-run
   campaign are untested.
6. **Structural**: with a dense `-distance` reward and a discount that makes the
   first action nearly irrelevant, the per-action signal sits near Monte-Carlo
   noise. Potential-based shaping on progress (`d(t) - d(t+1)`) would preserve the
   optimal policy while lifting the signal well above the noise — a reward-design
   change, not a tuning one.
