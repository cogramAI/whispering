FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# install python
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip

# install portaudio
RUN apt -y install portaudio19-dev

# torch
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Copy entire repo and install
COPY . /app

WORKDIR /app

RUN pip install .
