# Ubuntu 20.04 / ROS 2 Foxy / Python 3.8 - the same triple the experiments were
# recorded on. Do not move to a newer ROS: the pinned numba/numpy wheels are
# built for 3.8, and the recorded data would no longer be comparable.
FROM ros:foxy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# tf_transformations is imported directly by loopHandler_copy.py; the ROS
# message types it uses (LaserScan, Odometry, Twist) are already in ros-base.
#
# The X and GL libraries are dlopened by UnityPlayer.so at start-up, so they do
# not show up in ldd. They are only needed for --render window, but they cost
# about 60 MB and having them means one image covers both modes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-foxy-tf-transformations \
        python3-pip \
        libx11-6 libxcursor1 libxrandr2 libxinerama1 libxi6 libxext6 \
        libxrender1 libxxf86vm1 libxfixes3 \
        libgl1 libglx-mesa0 libglu1-mesa libegl1 libgbm1 \
        libasound2 \
        procps psmisc \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# ffmpeg is not optional: debug_utils.py writes the rollout animation as .mp4,
# and without it matplotlib falls back to the Pillow writer, which cannot
# encode mp4 and takes down the run *after* the episode has finished.

# pip 25 dropped Python 3.8. Pinning keeps the image buildable.
RUN python3 -m pip install --no-cache-dir --upgrade "pip==24.0" "setuptools<70" wheel

# Copied on its own so that editing the source does not invalidate this layer -
# it is by far the slowest one to rebuild.
COPY mctsVoRos/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Run as the invoking user so that the CSVs, pickles and plots written into the
# bind-mounted repo are owned by them rather than by root. video/render are for
# /dev/dri in windowed mode.
ARG UID=1000
ARG GID=1000
ARG VIDEO_GID=44
ARG RENDER_GID=109
RUN if ! getent group ${GID} >/dev/null; then groupadd -g ${GID} mctsvo; fi \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash mctsvo \
    && groupmod -g ${VIDEO_GID} video 2>/dev/null || true \
    && (getent group ${RENDER_GID} >/dev/null || groupadd -g ${RENDER_GID} render) \
    && usermod -aG ${VIDEO_GID},${RENDER_GID} mctsvo

# Plots and animations are always written to disk, never shown, and matplotlib
# picks a GUI backend when DISPLAY is set - which it is in windowed mode.
ENV MPLBACKEND=Agg
# Numba caches compiled kernels next to the source by default; the source is a
# read-only-ish bind mount shared with the host, so give it somewhere of its own.
ENV NUMBA_CACHE_DIR=/home/mctsvo/.numba_cache

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

USER mctsvo
WORKDIR /ws/src/MCTS_VO_ROS/mctsVoRos

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
