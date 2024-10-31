# Use Python 3.11-slim as base image
FROM python:3.11-slim

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Install CUDA dependencies (required if using NVIDIA GPUs with CUDA)
    cuda-command-line-tools-11-8 \
    libcudnn8 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port 4000 to bind Gunicorn correctly
EXPOSE 4000

# Command to run your application using Gunicorn, binding it to 0.0.0.0:4000
CMD ["gunicorn", "-w", "1", "--threads", "1", "-b", "0.0.0.0:4000", "app:app"]
