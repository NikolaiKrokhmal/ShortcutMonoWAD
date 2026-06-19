#!/bin/bash

IMAGE_NAME="ai_ev"
CONTAINER_NAME="ai_ev"
JUPYTER_PORT=8888

docker run --gpus all -it --rm \
    --name "$CONTAINER_NAME" \
    --shm-size=8g \
    -v "$(pwd)":/workspace \
    -v /mnt/c/Niko/Uni/Masters/courses/ai_ev/object:/workspace/data/KITTI/object \
    -p "$JUPYTER_PORT":8888 \
    "$IMAGE_NAME" \
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/workspace
