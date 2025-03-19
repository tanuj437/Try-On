# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

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
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install required Python packages
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY . /app

# Expose port
EXPOSE 7860

# Run the handler on serverless
CMD ["python3", "rp_handler.py"]
