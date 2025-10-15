FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-lc"]

RUN apt-get update -y --fix-missing && \
    apt-get install -y git build-essential curl wget ffmpeg unzip git-lfs sox libsox-dev && \
    apt-get clean && git lfs install

# Miniforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p /opt/conda && rm ~/miniforge.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda config --add channels conda-forge && conda config --set channel_priority strict
RUN conda create -y -n ${VENV} python=3.10
ENV PATH="/opt/conda/envs/${VENV}/bin:/opt/conda/bin:${PATH}"

# Optional extras
RUN conda run -n ${VENV} conda install -y -c conda-forge pynini==2.1.5

# Work in /code (this will be bind-mounted at runtime)
WORKDIR /code
ENV PYTHONPATH="/code:/code/third_party/Matcha-TTS"

# Install deps when the code is mounted
# (requirements.txt must exist in the mounted repo)
RUN echo '#!/bin/bash\nset -e\npip install -r requirements.txt --no-cache-dir' > /usr/local/bin/install_deps && \
    chmod +x /usr/local/bin/install_deps
