# Sample usage:
# docker build -t top -f aws.Dockerfile
# or, docker build -t top .
# docker run -p 6006:6006/ --volume /data:/home/user --shm-size 32G -it top

# FROM nvidia/cuda:11.0-base
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

RUN apt-get update && apt-get -y install sudo tmux

# Configure environment
ENV USER="user"
ENV HOME_DIR="/home/${USER}"
ENV WORK_DIR="${HOME_DIR}/"
ENV PATH="${HOME_DIR}/.local/bin:${PATH}"

# Prepare non root env with sudo access and no password
# TODO(ycho): Check if `--create-home` should be disabled perhaps?
RUN useradd --create-home --shell /bin/bash ${USER}
RUN usermod -append --groups sudo ${USER}
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${USER}
WORKDIR ${WORK_DIR}
