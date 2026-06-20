FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX" \
    WANDB_MODE=offline

RUN pip install --no-cache-dir \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY visualDet3D/networks/lib/ops ./visualDet3D/networks/lib/ops
RUN (cd visualDet3D/networks/lib/ops/dcn && FORCE_CUDA=1 python3 setup.py build_ext --inplace && rm -rf build) && \
    (cd visualDet3D/networks/lib/ops/iou3d && FORCE_CUDA=1 python3 setup.py build_ext --inplace && rm -rf build)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
