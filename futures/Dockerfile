# REDPAN Docker Image
# High-Performance SeisBench-Inspired RED-PAN Implementation
#
# This Dockerfile creates a containerized environment for REDPAN
# with all necessary dependencies for seismic phase picking.

FROM tensorflow/tensorflow:2.12.0-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NUMEXPR_MAX_THREADS=32

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    pkg-config \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core scientific computing packages
RUN pip install --no-cache-dir \
    numpy>=1.18.0,<2.0.0 \
    scipy>=1.7.0 \
    pandas>=1.3.0,<3.0.0 \
    scikit-learn>=0.24.0 \
    h5py>=3.0.0

# Install seismology packages
RUN pip install --no-cache-dir \
    obspy>=1.2.0 \
    matplotlib>=3.0.0 \
    seaborn>=0.11.0

# Install performance and multiprocessing packages
RUN pip install --no-cache-dir \
    psutil>=5.0.0 \
    multiprocess>=0.70.0 \
    numba>=0.50.0

# Install development and testing tools
RUN pip install --no-cache-dir \
    pytest>=6.0.0 \
    pytest-cov>=2.0.0 \
    pytest-xdist>=2.0.0 \
    black>=21.0.0 \
    flake8>=3.8.0 \
    jupyter>=1.0.0 \
    ipython>=7.0.0

# Copy REDPAN source code
COPY . /app/

# Install REDPAN
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /data /models /output /workspace

# Set up environment for TensorFlow GPU
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Create non-root user for security
RUN useradd -m -u 1000 redpan && \
    chown -R redpan:redpan /app /data /models /output /workspace

USER redpan

# Expose ports for Jupyter notebook
EXPOSE 8888

# Set entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER redpan

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["bash"]

# Metadata
LABEL maintainer="REDPAN Team <redpan@example.com>"
LABEL version="1.0.0"
LABEL description="High-Performance SeisBench-Inspired RED-PAN Implementation"
LABEL org.opencontainers.image.source="https://github.com/tso1257771/RED-PAN"
LABEL org.opencontainers.image.documentation="https://github.com/tso1257771/RED-PAN/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import redpan; print('REDPAN import successful')" || exit 1

# Volume mounts for data, models, and output
VOLUME ["/data", "/models", "/output", "/workspace"]
