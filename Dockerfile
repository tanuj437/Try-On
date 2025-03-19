# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /TRY-ON
WORKDIR /TRY-ON

# Copy all files from the current directory to the container
COPY . /TRY-ON

# Upgrade pip and install required Python packages
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 7860 for Gradio app
EXPOSE 7860

# Run rp_handler.py for serverless on startup
CMD ["python3", "rp_handler.py"]
