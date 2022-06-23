# Get latest pytorch/cuda docker image from nvidia
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common

RUN apt-get install -y --no-install-recommends pkg-config
RUN apt install -y --no-install-recommends ffmpeg

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    git

RUN pip install --upgrade setuptools
RUN pip install \
    pillow \
    pycocotools \
    wheel \
    ifcfg \
    einops \
    wandb \
    cython \    
    scikit-learn \
    opencv-python \
    numba \
    progress \
    matplotlib \
    easydict \
    scipy \
    pyquaternion \
    nuscenes-devkit \
    pyyaml \
    motmetrics==1.1.3 \
    tensorboardx
  
RUN pip install git+https://github.com/lbin/DCNv2.git@pytorch_1.9


