# Running the experiments in Docker

Runs the experiments without a system ROS install or the `venv`. The image
carries ROS 2 Foxy, Python 3.8 and the pinned scientific stack; the repo and the
Unity builds are mounted from the host, so your edits apply immediately and
results land in `mctsVoRos/debug/` owned by you.

## Setup

```bash
cd ~/colcon_ws/src/MCTS_VO_ROS
docker/run.sh --build          # first time only, a few minutes
```

Rebuild with the same command after changing the `Dockerfile` or
`requirements.txt`. Nothing else needs a rebuild.

## Running things

The working directory inside the container is `mctsVoRos/`, the same place you
run these commands on the host, so anything that works there works here by
prefixing `docker/run.sh`:

```bash
# one run
docker/run.sh python3 loopHandler_copy.py \
    --algorithm VO-TREE --trajectories intention --exp_num 0 \
    --env-render headless

# a campaign
docker/run.sh ./run_all_experiments.sh -n 30 --skip-setup

# analysis
docker/run.sh python3 summarize_debug.py --no-csv --label B4
docker/run.sh python3 summarize_debug.py plot --label B4 --anim

# poke around
docker/run.sh
```

Pass `--skip-setup` to `run_all_experiments.sh`. Without it the script tries to
source ROS, run `colcon build` and activate `venv/`, none of which exist in the
container.

Output goes to the usual places on the host: `mctsVoRos/debug/`,
`debug_archive/`, `debug_plots/`, `analysis/`.

## Choosing a render mode

`--env-render headless` needs no display and is the one to use for anything you
intend to compare or leave running unattended.

`--env-render window` opens a real Unity window on your desktop. It works out of
the box — `run.sh` passes the X socket, `$XAUTHORITY` and `/dev/dri` through
whenever `DISPLAY` is set. Over ssh without X forwarding there is no display, so
use headless.

Rendering inside the container is software (llvmpipe), giving roughly 16 FPS.
That is fine for watching a run, and on the default `fixed` build it does not
affect the experiment.

## Two things to watch

**Reproduce the published configuration on the host, not here.** On
`--env-build orig` and `50hz` the obstacles step on the frame clock, so their
speed follows the frame rate — and the container's frame rate is not the host's.
Those builds are only approximately transferable. The default `fixed` build is
immune to this and runs identically in both places.

**Runs are not reproducible run to run**, in the container or on the host. The
planner's search budget is wall-clock, so the same `--exp_num` gives a different
trajectory each time. Do not read a difference at n=5 as real without a test.

## Parallel runs

Safe. Each container gets its own network namespace, and Unity talks to the
planner over DDS inside it, so concurrent runs cannot see or disturb each other.
They do compete for CPU, which shows up as fewer simulations per step.

Use `--net-host` only if you want to inspect the topics from outside; it gives
up that isolation.

## What it does not do

- **Build the Unity environments.** No Unity Editor in the image. Rebuild on the
  host with `Assets/Editor/BuildScript.cs`.
- **Switch the nested `MCTS_VO` repo.** Whichever commit is checked out on the
  host is what runs; changing branches still needs
  `git -C mctsVoRos/MCTS_VO checkout <sha>`.
- **`colcon build`.** Nothing here needs it; the loop runs as a plain script.

## If something goes wrong

**`permission denied ... /var/run/docker.sock`** — you are in the `docker` group
but this login session predates it. Log out and back in, or for now:

```bash
sg docker -c "docker/run.sh ..."
```

**`the repo does not appear to be mounted`** — you ran `docker run` directly
rather than through `docker/run.sh`, which is what sets up the mounts.

**Output files owned by root** — the image was built by a different user. The
UID is baked in at build time, so rebuild with `docker/run.sh --build`.

**A run leaves a Unity process behind** — `run.sh` starts the container with
`--init`, so killing it takes the simulator with it. If one does survive:
`pkill -f 'env_build/.*env.x86_64'`.
