# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /TRY-ON
WORKDIR /TRY-ON

# Copy all files to the container
COPY . /TRY-ON

# Convert files to Unix format
RUN dos2unix /TRY-ON/requirements.txt

# Upgrade pip and install dependencies in chunks to debug
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Expose port 7860 for Gradio app
EXPOSE 7860

# Run rp_handler.py for serverless
CMD ["python3", "rp_handler.py"]
