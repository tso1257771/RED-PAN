# REDPAN Docker Guide

This guide covers how to use REDPAN with Docker for containerized seismic processing.

## Quick Start

### Build the Image

```bash
# Build REDPAN Docker image
docker build -t redpan:latest .

# Or using docker-compose
docker-compose build
```

### Run REDPAN Demo

```bash
# Run demo with your data
docker run --gpus all \
  -v /path/to/waveforms:/data \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  redpan:latest demo

# Or using docker-compose
docker-compose run redpan-demo
```

## Available Commands

### Demo Processing
```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  redpan:latest demo
```

### Parallel Processing
```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  redpan:latest parallel --processes 8 --batch-size 32
```

### Jupyter Notebook
```bash
docker run --gpus all -p 8888:8888 \
  -v /path/to/workspace:/workspace \
  redpan:latest jupyter

# Access at http://localhost:8888
```

### Interactive Shell
```bash
docker run --gpus all -it \
  -v /path/to/data:/data \
  redpan:latest bash
```

### Run Tests
```bash
docker run --gpus all redpan:latest test
```

### Run Benchmarks
```bash
docker run --gpus all redpan:latest benchmark
```

## Volume Mounts

The container expects these volume mounts:

- `/data` - Input waveform data directory
- `/models` - Pre-trained model files
- `/output` - Output picks directory
- `/workspace` - Jupyter workspace (optional)

## Docker Compose

For easier development and deployment:

```yaml
# docker-compose.yml
version: '3.8'

services:
  redpan:
    image: redpan:latest
    volumes:
      - ./data:/data
      - ./models:/models
      - ./output:/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Usage with Docker Compose

```bash
# Start all services
docker-compose up

# Run specific service
docker-compose run redpan-demo

# Start Jupyter
docker-compose up redpan-jupyter

# Run tests
docker-compose run redpan-test
```

## GPU Support

### NVIDIA Docker Setup

1. Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Test GPU access:
```bash
docker run --gpus all nvidia/cuda:11.2-base-ubuntu20.04 nvidia-smi
```

### Using GPU in Container

```bash
# Single GPU
docker run --gpus all redpan:latest

# Specific GPU
docker run --gpus '"device=0"' redpan:latest

# Multiple GPUs
docker run --gpus 2 redpan:latest
```

## Environment Variables

Configure the container behavior:

```bash
docker run \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NUMEXPR_MAX_THREADS=16 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  redpan:latest
```

Available variables:
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `NUMEXPR_MAX_THREADS` - NumPy thread count
- `TF_FORCE_GPU_ALLOW_GROWTH` - TensorFlow GPU memory growth

## Data Organization

Organize your data for container processing:

```
project/
├── data/           # Mount to /data
│   ├── 2019.188.08/
│   │   ├── PB.STA1.HHZ.00.sac
│   │   ├── PB.STA1.HHN.00.sac
│   │   └── PB.STA1.HHE.00.sac
│   └── 2019.189.08/
├── models/         # Mount to /models  
│   ├── train.hdf5
│   └── train.json
├── output/         # Mount to /output
└── workspace/      # Mount to /workspace (for Jupyter)
```

## Performance Optimization

### CPU Optimization
```bash
# Limit CPU cores
docker run --cpus="8.0" redpan:latest

# Set CPU affinity
docker run --cpuset-cpus="0-7" redpan:latest
```

### Memory Optimization
```bash
# Limit memory
docker run --memory="8g" redpan:latest

# Enable swap accounting
docker run --memory="8g" --memory-swap="12g" redpan:latest
```

### Batch Processing
```bash
# Optimize batch size for your GPU
docker run --gpus all redpan:latest parallel --batch-size 64

# Multiple processes for CPU processing
docker run redpan:latest parallel --processes 16
```

## Development Workflow

### Development Container
```bash
# Mount source code for development
docker run -it \
  -v $(pwd):/app \
  -v /path/to/data:/data \
  --gpus all \
  redpan:latest bash

# Install in development mode
pip install -e .
```

### Code Formatting
```bash
docker run -v $(pwd):/app redpan:latest black redpan/
docker run -v $(pwd):/app redpan:latest flake8 redpan/
```

### Testing in Container
```bash
# Run all tests
docker run redpan:latest test

# Run specific tests
docker run redpan:latest python -m pytest tests/test_redpan.py -v

# Run with coverage
docker run redpan:latest python -m pytest tests/ --cov=redpan
```

## Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Check Docker GPU support
   docker run --gpus all nvidia/cuda:11.2-base-ubuntu20.04 nvidia-smi
   ```

2. **Out of memory**:
   ```bash
   # Reduce batch size
   docker run redpan:latest demo --batch-size 8
   
   # Monitor memory
   docker stats
   ```

3. **Permission errors**:
   ```bash
   # Fix ownership
   sudo chown -R $(id -u):$(id -g) ./output
   
   # Run as current user
   docker run -u $(id -u):$(id -g) redpan:latest
   ```

### Debugging

```bash
# Check container logs
docker logs redpan-container

# Interactive debugging
docker run -it --entrypoint bash redpan:latest

# Check TensorFlow installation
docker run redpan:latest python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

## Production Deployment

### Multi-Stage Build (Smaller Image)

```dockerfile
# Multi-stage Dockerfile for production
FROM tensorflow/tensorflow:2.12.0-gpu as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM tensorflow/tensorflow:2.12.0-gpu
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
RUN pip install -e .
```

### Health Checks

```bash
# Check container health
docker run --health-cmd="python -c 'import redpan'" redpan:latest

# Monitor health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Resource Limits

```yaml
# docker-compose.yml with resource limits
services:
  redpan:
    image: redpan:latest
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Support

For issues with Docker deployment:

1. Check the [main README](../README.md) for general REDPAN usage
2. Review [troubleshooting section](#troubleshooting)
3. Open an issue on [GitHub](https://github.com/tso1257771/RED-PAN/issues)

## Examples

See the [examples](../examples/) directory for complete Docker workflows and use cases.
