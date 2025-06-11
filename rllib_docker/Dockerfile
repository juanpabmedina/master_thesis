# Base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Ray with RLlib
RUN pip install --upgrade pip
RUN pip install "ray[rllib]" torch "gymnasium[atari,accept-rom-license,mujoco]"

# Optional: copy your project files
COPY . /app

# Default command
CMD [ "bash" ]
