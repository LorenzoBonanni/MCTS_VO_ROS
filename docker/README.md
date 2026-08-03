# Running the experiments in Docker

A container image carrying ROS 2 Foxy, Python 3.8 and the pinned scientific
stack, so the experiments run without a system ROS install or the `venv`. The
repo and the Unity builds are bind-mounted rather than copied, so code edits
take effect immediately and results are written straight into `mctsVoRos/debug/`
on the host, owned by you.

## Quick start

```bash
cd ~/colcon_ws/src/MCTS_VO_ROS

docker/run.sh --build                       # first time only, a few minutes

docker/run.sh python3 loopHandler_copy.py \
    --algorithm VO-TREE --trajectories intention --exp_num 0 \
    --env-render headless
```

The working directory inside the container is `mctsVoRos/`, the same place you
run these commands on the host, so **every command that works on the host works
here unchanged** — including the campaign script and the analysis tooling:

```bash
docker/run.sh ./run_all_experiments.sh -n 30 --skip-setup
docker/run.sh python3 summarize_debug.py --no-csv
docker/run.sh                                # interactive shell
```

`--skip-setup` is required for the campaign script here: it otherwise sources
ROS, runs `colcon build` and activates `venv/`, none of which apply inside the
container — the entrypoint has already sourced ROS, and the requirements are
installed into the system interpreter. Verified end to end with it.

### If you get "permission denied ... docker.sock"

You are in the `docker` group but the login session predates it. Either log out
and back in, or prefix commands for this session:

```bash
sg docker -c "docker/run.sh ..."
```

## Headless and windowed both work

`run.sh` passes the X socket, `$XAUTHORITY` and `/dev/dri` through whenever
`DISPLAY` is set, so `--env-render low` and `--env-render full` open a real
Unity window on your desktop. Nothing extra is needed; with no `DISPLAY` (over
ssh, say) the script says so and `--env-render headless`, the default, still
works — verified with `DISPLAY` and `XAUTHORITY` unset, so no X socket and no
`/dev/dri` were mounted at all.

Verified on this machine, one full VO-TREE run on the intention scene:

| | outcome | steps | sims/step |
|---|---|---|---|
| headless, no X at all | goal reached | 278 | 49.2 |
| headless, X available | obsCollision | 293 | 41.2 |
| windowed | goal reached | 255 | 38.8 |

All three wrote the CSV, the trajectory GIF and the rollout MP4. The spread in
sims/step is the wall-clock search budget, not the container; the host varies
the same way.

**On this branch the render mode is not cosmetic.** `LaserScanner2D` scans in
`Update()`, so the publish rate is clamped to the frame rate: 16.8 Hz at
`full`, 39.4 Hz at `low`, 49.7 Hz headless, and `control_loop` skips a tick
whenever no new scan arrived. Choosing a render mode therefore chooses a
control cycle. Since the container's frame rate is not the host's (below), a
windowed run here is not interchangeable with a windowed run on the host —
which is a reason to prefer `headless` for anything you intend to compare.

## What is and is not reproduced exactly

**Obstacle speed is reproduced.** Measured with `ObstacleProbe` over 20 s on the
`fixed` build, windowed:

| | frame rate | Obstacle_9 | Obstacle_10 |
|---|---|---|---|
| host | 17.2 FPS | 0.0991 m/s | 0.0994 m/s |
| container | 16.2 FPS | 0.0992 m/s | 0.0993 m/s |

The frame rates differ by about 6% and the obstacle speeds agree to 0.1%, which
is the whole point of the `c01fff6` fix — on the default `*_env_fixed` builds
the environment no longer depends on how fast Unity renders.

**On the pre-fix builds it is not**, and this is the one thing to watch. Those
are the ones reached by passing a path to `--env-build`:

```
../env_build/{sin,int}_env/env.x86_64        pre-fix, 7.5 Hz sensors
../env_build/{sin,int}_env_50hz/env.x86_64   pre-fix, 50 Hz sensors
```

There the movement scripts still step on the frame clock, so the 6% frame-rate
gap becomes a 6% obstacle-speed gap, and a windowed run of one of those in the
container is a slightly different experiment from the same run on the host.
Rendering inside the container goes through llvmpipe (software Mesa), and a
machine with different graphics will differ by more than 6%. Reproduce the
published configuration on the host, or accept that a windowed pre-fix run is
only approximately transferable.

Two smaller notes. The container renders through llvmpipe even though
`/dev/dri` is passed in, because this host's virtio GPU exposes no accelerated
GL; that costs frame rate, not correctness. And runs are not reproducible
run-to-run in either environment — the planner's search budget is wall-clock,
so identical inputs give different trajectories. That is a property of the code,
not of the container.

## How it is put together

`Dockerfile` starts from `ros:foxy-ros-base` — the same Ubuntu 20.04 / Foxy /
Python 3.8 triple the results were recorded on. Do not move it to a newer ROS:
the pinned `numba` and `numpy` wheels are built for 3.8, and the recorded data
would stop being comparable.

On top of that it installs `ros-foxy-tf-transformations` (imported directly by
`loopHandler_copy.py`), the pinned `requirements.txt`, the X and GL libraries
`UnityPlayer.so` dlopens, and **ffmpeg** — without which the rollout `.mp4`
cannot be encoded and the run dies *after* the episode has finished, taking the
animation with it.

The image builds a user with your UID and GID so that everything written into
the bind mount is yours rather than root's. This is why `run.sh --build` passes
`--build-arg UID=...`: an image built for one user is wrong for another.

Unity and the planner both run inside the one container and find each other over
DDS — the Unity build bundles its own CycloneDDS, which is why no
`ros_tcp_endpoint` is involved. Because the container has its own network
namespace by default, that DDS traffic never reaches the host and two containers
cannot interfere with each other, so parallel runs are safe. `--net-host` is
available if you ever need to see the topics from outside, at the cost of that
isolation.

There is deliberately no `docker-compose.yml`: the UID and X11 plumbing are
computed at run time, and a compose file would either duplicate that logic or
drift out of step with it.

## Things the container does not solve

- **Building the Unity environments.** The image runs the compiled
  `env_build/*/env.x86_64`; it has no Unity Editor. Rebuild on the host with
  `Assets/Editor/BuildScript.cs` as before.
- **The nested `MCTS_VO` repo.** It is bind-mounted along with everything else,
  so whichever commit is checked out on the host is the one that runs. Switching
  branches still needs the separate `git -C mctsVoRos/MCTS_VO checkout`.
- **`colcon build`.** Nothing here needs it — the loop is run as a plain script,
  not as a ROS entry point.
