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

## Building and running without the script

`run.sh` is only a wrapper. The build it runs is:

```bash
cd ~/colcon_ws/src/MCTS_VO_ROS          # context must be the repo root,
                                        # the Dockerfile copies
                                        # mctsVoRos/requirements.txt
docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g "$(id -un)")" \
    --build-arg VIDEO_GID="$(getent group video  | cut -d: -f3)" \
    --build-arg RENDER_GID="$(getent group render | cut -d: -f3)" \
    -f docker/Dockerfile -t mctsvo:foxy .
```

Every argument has a default (`1000`, `1000`, `44`, `109`), so a bare
`docker build -f docker/Dockerfile -t mctsvo:foxy .` works too — it just gives
root-owned output files if your UID is not 1000, and no access to `/dev/dri` if
the video and render groups differ.

Running one headless experiment:

```bash
docker run --rm --init \
    -v "$PWD:/ws/src/MCTS_VO_ROS" \
    -w /ws/src/MCTS_VO_ROS/mctsVoRos \
    mctsvo:foxy \
    python3 loopHandler_copy.py --algorithm VO-TREE \
        --trajectories intention --exp_num 0 --env-render headless
```

The mount is the only part that is not optional — the image contains no source
and no Unity build, and the entrypoint refuses to start without it. For a
window, add `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --device /dev/dri`.

### On an HPC cluster (Podman, e.g. CSCS)

Podman takes the same `Dockerfile` and the same arguments. Following
[the CSCS instructions](https://docs.cscs.ch/build-install/containers/):

```bash
# once: tell Podman to build in /dev/shm rather than $HOME
mkdir -p "$HOME/.config/containers"
cat > "$HOME/.config/containers/storage.conf" <<'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
```

Build and import in **one allocation** — `/dev/shm` does not survive the job,
so a build in one job cannot be imported from another:

```bash
srun --pty --partition=<partition> bash

cd $SCRATCH/MCTS_VO_ROS
podman build --network=host \
    --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    -f docker/Dockerfile -t mctsvo:foxy .

enroot import -x mount -o $SCRATCH/mctsvo.sqsh podman://mctsvo:foxy
```

Then an EDF at `$HOME/.edf/mctsvo.toml`:

```toml
image = "/capstor/scratch/cscs/<user>/mctsvo.sqsh"
mounts = ["/capstor/scratch/cscs/<user>:/capstor/scratch/cscs/<user>"]
workdir = "/capstor/scratch/cscs/<user>/MCTS_VO_ROS/mctsVoRos"
entrypoint = true

[env]
NUMBA_CACHE_DIR = "/capstor/scratch/cscs/<user>/.numba"
MPLCONFIGDIR = "/capstor/scratch/cscs/<user>/.mpl"
```

```bash
srun --environment=mctsvo python3 loopHandler_copy.py \
    --algorithm VO-TREE --trajectories intention --exp_num 0 \
    --env-render headless
```

Four things differ from a workstation, and they are what will bite:

- **`entrypoint = true` matters.** The container engine does not run the image
  entrypoint unless asked, and the entrypoint is what sources ROS. Without it
  `import rclpy` fails; the equivalent is to wrap each command in
  `bash -c 'source /opt/ros/foxy/setup.bash && ...'`.
- **`$HOME` inside the container is not yours.** The engine runs as your cluster
  UID, not the image's `mctsvo` user, so `/home/mctsvo` is not writable — hence
  the `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` overrides above. Without them numba
  recompiles every run and matplotlib complains.
- **Headless only.** There is no display and no `/dev/dri`, so
  `--env-render headless` is the only mode. It needs no GPU.
- **Clone to scratch, not `$HOME`.** The repo carries ~890 MB of Unity builds,
  and the mount in the EDF has to cover wherever you put it.

Because the Unity builds are committed, a clone is all that is needed — there is
nothing to compile on the cluster and no Unity Editor involved.

## Running things

The working directory inside the container is `mctsVoRos/`, the same place you
run these commands on the host, so anything that works there works here by
prefixing `docker/run.sh`:

```bash
# one run
docker/run.sh python3 loopHandler_copy.py \
    --algorithm VO-TREE --trajectories intention --exp_num 0

# a campaign
docker/run.sh ./run_all_experiments.sh -n 30 --skip-setup

# analysis
docker/run.sh python3 summarize_debug.py --no-csv
docker/run.sh python3 summarize_debug.py plot --anim

# poke around
docker/run.sh
```

Pass `--skip-setup` to `run_all_experiments.sh`. Without it the script tries to
source ROS, run `colcon build` and activate `venv/`, none of which exist in the
container.

Output goes to the usual places on the host: `mctsVoRos/debug/`,
`debug_archive/`, `debug_plots/`, `analysis/`. Use `--suffix` to keep sweeps
from overwriting each other, exactly as on the host.

## Choosing a render mode

`--env-render` is not a cosmetic choice on this branch: the scanner runs in
`Update()`, so the sensor publish rate is clamped to the frame rate, and the
control loop skips a tick whenever no new scan has arrived. The render mode
therefore sets the control cycle.

- `headless` (the default) — no display needed. Use it for anything you intend
  to compare, and for unattended campaigns.
- `low` — watchable, opens a window.
- `full` — Ultra with vSync, the slowest.

Windowed modes work out of the box: `run.sh` passes the X socket, `$XAUTHORITY`
and `/dev/dri` through whenever `DISPLAY` is set. Over ssh without X forwarding
there is no display, so use `headless`.

Rendering inside the container is software (llvmpipe), so its frame rate — and
hence the sensor rate under `low` and `full` — is not the host's. One more
reason to keep `headless` for comparable work.

## Two things to watch

**Reproduce the published configuration on the host, not here.** The pre-fix
builds, reached by passing a path to `--env-build`:

```
../env_build/{sin,int}_env/env.x86_64        pre-fix, 7.5 Hz sensors
../env_build/{sin,int}_env_50hz/env.x86_64   pre-fix, 50 Hz sensors
```

have obstacles that step on the frame clock, so their speed follows the frame
rate, which differs between container and host. Those runs are only
approximately transferable. The default `*_env_fixed` builds are immune and run
identically in both places.

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
