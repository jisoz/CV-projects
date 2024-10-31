# Use Python 3.11-slim as base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


# Install necessary system packages
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
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
