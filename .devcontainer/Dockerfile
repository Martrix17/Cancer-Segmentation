# Base PyTorch image
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# System dependencies
RUN apt-get update && apt-get install -y \
    git cmake clang-format wget unzip vim less \
    build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-commit hooks
RUN pip install pre-commit

# Create non-root developer user
RUN useradd -m developer && chown -R developer /workspace
USER developer

# Environment variables
ENV PYTHONPATH=/workspace
