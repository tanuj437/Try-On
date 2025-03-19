# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH /opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Set the working directory
WORKDIR /TRY-ON

# Copy requirements first
COPY requirements.txt .
RUN dos2unix requirements.txt

# Create and activate Conda environment
RUN conda create -n leffa python=3.10 -y
SHELL ["conda", "run", "-n", "leffa", "/bin/bash", "-c"]

# Install RunPod
RUN pip install runpod

# Copy the entire project
COPY . .

# Handle line ending issues
RUN find /TRY-ON -type f -name "*.py" -exec dos2unix {} \; || true

# Install requirements
RUN pip install -r requirements.txt

# Create cache directories
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.cache/torch

# Port for RunPod's health checks
EXPOSE 8000

# Set the entrypoint to run with conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "leffa", "python", "-u", "rp_handler.py"]