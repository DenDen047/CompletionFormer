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
RUN pip install -U openmim
RUN pip install mmcv==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# Install MMSegmentation v0.22.1
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
RUN git reset --hard e518d25e731be97aa1da704df83542f2951bdd21
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install -e .

# Install MMEngine
RUN pip install mmengine

# Install NVIDIA Apex
WORKDIR /
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /apex
RUN git reset --hard a78ccf0b3e3f7130b3f157732dc8e8e651389922
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# other packages
RUN pip install timm tqdm thop tensorboardX opencv-python ipdb h5py ipython Pillow==9.5.0 setuptools==59.5.0

# Install CompletionFormer
ADD . /CompletionFormer
RUN ls -l /CompletionFormer
# RUN git clone https://github.com/DenDen047/CompletionFormer.git /CompletionFormer
WORKDIR /CompletionFormer/src/model/deformconv
ENV CUDA_HOME /usr/local/cuda-11.3
RUN bash make.sh

RUN pip install tensorboard


WORKDIR /workspace