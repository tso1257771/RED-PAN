FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
# FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash redpan

# Set working directory
WORKDIR /home/redpan/app

# Copy requirements
COPY requirements.txt .

# Install Python packages directly (NGC image already has TensorFlow)
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt


# Change ownership of the directory
RUN mkdir -p /home/redpan/app/RED-PAN/outputs && \ 
    mkdir -p /home/redpan/app/RED-PAN/outputs_hr && \
    chown -R redpan:redpan /home/redpan && \
    chmod -R 777 /home/redpan

# Switch to non-root user
# USER redpan

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV BOOTSTRAP_SERVERS=kafka:9092

CMD ["/bin/bash"]