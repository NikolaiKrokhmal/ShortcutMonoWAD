FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

WORKDIR /workspace

RUN pip install --no-cache-dir \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN set -e && \
    export CUDA_HOME=/usr/local/cuda && \
    export FORCE_CUDA=1 && \
    cd visualDet3D/networks/lib/ops/dcn && \
    python3 setup.py build_ext --inplace && \
    rm -rf build && \
    cd /workspace && \
    cd visualDet3D/networks/lib/ops/iou3d && \
    python3 setup.py build_ext --inplace && \
    rm -rf build

CMD ["/bin/bash"]
