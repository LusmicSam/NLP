# Use a basic Ubuntu image
FROM ubuntu:22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git

# Install the Python libraries
# We install llama-cpp-python and psutil for monitoring
RUN pip3 install llama-cpp-python psutil

# Set a working directory inside the container
WORKDIR /app

# Copy your local benchmarking script (which we'll create next)
# into the container's /app directory
COPY benchmark.py /app/benchmark.py

# Default command to keep the container running
CMD ["/bin/bash"]