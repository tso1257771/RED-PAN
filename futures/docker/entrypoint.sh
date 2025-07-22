#!/bin/bash
# REDPAN Docker Entrypoint Script

set -e

# Configure TensorFlow GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set up logging
echo "🐳 Starting REDPAN Docker Container"
echo "=================================="
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "REDPAN version: $(python -c 'import redpan; print(redpan.__version__)')"
echo "Python version: $(python --version)"
echo "Available GPUs: $(python -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices(\"GPU\")))')"
echo "=================================="

# Function to show usage
show_usage() {
    echo "REDPAN Docker Container Usage:"
    echo ""
    echo "Available commands:"
    echo "  demo      - Run REDPAN demonstration"
    echo "  parallel  - Run parallel processing demo"
    echo "  test      - Run test suite"
    echo "  benchmark - Run performance benchmarks"
    echo "  jupyter   - Start Jupyter notebook server"
    echo "  bash      - Start interactive bash shell"
    echo ""
    echo "Examples:"
    echo "  docker run --gpus all -v /path/to/data:/data redpan demo"
    echo "  docker run --gpus all -v /path/to/data:/data -v /path/to/models:/models redpan parallel"
    echo "  docker run --gpus all -p 8888:8888 redpan jupyter"
    echo ""
}

# Handle different commands
case "$1" in
    demo)
        echo "🚀 Running REDPAN Demo"
        shift
        if [ "$#" -eq 0 ]; then
            echo "Usage: docker run --gpus all -v /data:/data -v /models:/models -v /output:/output redpan demo"
            echo "Required volumes:"
            echo "  -v /path/to/waveforms:/data"
            echo "  -v /path/to/models:/models" 
            echo "  -v /path/to/output:/output"
            exit 1
        fi
        exec redpan-demo --data-dir /data --output-dir /output --model /models/train.hdf5 "$@"
        ;;
    
    parallel)
        echo "⚡ Running REDPAN Parallel Processing"
        shift
        if [ "$#" -eq 0 ]; then
            echo "Usage: docker run --gpus all -v /data:/data -v /models:/models -v /output:/output redpan parallel"
            echo "Optional arguments: --processes N --batch-size N"
            echo "Required volumes:"
            echo "  -v /path/to/waveforms:/data"
            echo "  -v /path/to/models:/models"
            echo "  -v /path/to/output:/output"
            exit 1
        fi
        exec redpan-parallel --data-dir /data --output-dir /output --model /models/train.hdf5 "$@"
        ;;
    
    test)
        echo "🧪 Running REDPAN Test Suite"
        shift
        exec redpan-test "$@"
        ;;
    
    benchmark)
        echo "📊 Running REDPAN Benchmarks"
        shift
        exec redpan-benchmark "$@"
        ;;
    
    jupyter)
        echo "📓 Starting Jupyter Notebook Server"
        shift
        cd /workspace
        exec jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token='' \
            --NotebookApp.password='' \
            --NotebookApp.allow_origin='*' \
            --NotebookApp.base_url=/ \
            "$@"
        ;;
    
    python)
        echo "🐍 Starting Python Interactive Session"
        shift
        exec python "$@"
        ;;
    
    bash|sh)
        echo "💻 Starting Interactive Shell"
        exec /bin/bash
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        if [ "$#" -eq 0 ]; then
            echo "ℹ️  No command specified, starting interactive shell"
            exec /bin/bash
        else
            echo "🔧 Executing custom command: $@"
            exec "$@"
        fi
        ;;
esac
