# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Make sure git is available in PATH
RUN which git

# Set the working directory to /TRY-ON to match your structure
WORKDIR /TRY-ON

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN dos2unix requirements.txt

# Install runpod first separately to ensure it's available
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir runpod

# Copy the entire project (respecting your directory structure)
# Do this BEFORE installing requirements to ensure Git repositories can be accessed
COPY . .

# Handle line ending issues
RUN find /TRY-ON -type f -name "*.py" -exec dos2unix {} \; || true
RUN find /TRY-ON -type f -name "*.sh" -exec dos2unix {} \; || true

# Now install requirements after copying the project
RUN pip3 install --no-cache-dir -r requirements.txt

# Set execution permissions for Python files
RUN chmod +x rp_handler.py

# Create cache directories for better model loading
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.cache/torch

# Port for RunPod's health checks
EXPOSE 8000

# RunPod handler as the entrypoint
CMD ["python3", "-u", "rp_handler.py"]