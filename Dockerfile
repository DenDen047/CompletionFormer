ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip

RUN conda clean --all

# Install MMCV
# RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install -U openmim
RUN mim install mmcv-full==1.4.4

# Install MMSegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
RUN git clone --depth 1 --branch v0.22.1 https://github.com/open-mmlab/mmsegmentation.git
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# RUN conda init
# RUN conda create -n completionformer python=3.8 -y
# RUN conda activate completionformer
# RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113
# RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1
# RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
# RUN pip install -U openmim
# RUN mim install mmcv-full==1.4.4
# RUN mim install mmsegmentation==0.22.1

RUN pip install timm tqdm thop tensorboardX opencv-python ipdb h5py ipython Pillow==9.5.0

# Install CompletionFormer
ADD . /CompletionFormer
RUN ls -l /CompletionFormer
# RUN git clone https://github.com/DenDen047/CompletionFormer.git /CompletionFormer
WORKDIR /CompletionFormer/src/model/deformconv
ENV CUDA_HOME /usr/local/cuda-11.3
RUN bash make.sh

WORKDIR /workspace
