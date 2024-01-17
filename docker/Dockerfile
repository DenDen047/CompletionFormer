FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# Python 3.7.11
# conda 4.10.3

# RUN conda init
# RUN conda create -n completionformer python=3.8 -y
# RUN conda activate completionformer
# RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113
RUN pip install -U pip
RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1
RUN pip install timm tqdm thop tensorboardX opencv-python ipdb h5py ipython Pillow==9.5.0

# Install CompletionFormer
WORKDIR /tmp
RUN git clone https://github.com/DenDen047/CompletionFormer.git
WORKDIR /tmp/CompletionFormer/src/model/deformconv
RUN bash make.sh