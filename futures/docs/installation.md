# Installation and Setup Guide

## Requirements

### System Requirements
- Python 3.7 or higher
- NumPy >= 1.18.0
- TensorFlow >= 2.4.0 (for model inference)
- Optional: ObsPy >= 1.2.0 (for seismic data handling)
- Optional: Matplotlib >= 3.0.0 (for visualization)

### Hardware Requirements
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB+ RAM, multi-core CPU
- **Optimal**: 16GB+ RAM, GPU with CUDA support

## Installation Methods

### Method 1: Direct Integration (Recommended)

Since REDPAN is designed as a drop-in replacement, you can integrate it directly into your existing RED-PAN setup:

1. **Copy the REDPAN directory** to your project:
   ```bash
   cp -r REDPAN/redpan /path/to/your/project/
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy tensorflow
   # Optional dependencies
   pip install obspy matplotlib
   ```

3. **Import and use**:
   ```python
   from redpan import create_picker
   
   picker = create_picker(model=your_model)
   predP, predS, predM = picker.predict(waveform)
   ```

### Method 2: Development Installation

For development and testing:

1. **Clone or copy the repository**:
   ```bash
   git clone /path/to/REDPAN
   cd REDPAN
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Run tests**:
   ```bash
   python tests/test_redpan.py
   python tests/benchmark_suite.py
   ```

### Method 3: Package Installation (Future)

When packaged for PyPI (future release):

```bash
pip install redpan
```

## Quick Setup Verification

Create a test script to verify your installation:

```python
#!/usr/bin/env python3
"""Quick setup verification for REDPAN"""

import numpy as np
from redpan import create_picker

# Create dummy model
class DummyModel:
    def predict(self, data, verbose=0):
        batch_size, pred_npts = data.shape[0], data.shape[1]
        predictions = np.random.random((batch_size, pred_npts, 2)) * 0.1
        masks = np.random.random((batch_size, pred_npts, 1)) * 0.1
        return predictions, masks

# Test REDPAN
model = DummyModel()
picker = create_picker(model=model)

# Create test waveform (5 minutes at 100 Hz)
waveform = np.random.normal(0, 0.1, 30000)

# Run prediction
predP, predS, predM = picker.predict(waveform)

print("REDPAN Setup Verification")
print("=" * 40)
print(f"✓ Successfully created picker")
print(f"✓ Input waveform: {len(waveform)} samples")
print(f"✓ P predictions: {predP.shape}")
print(f"✓ S predictions: {predS.shape}") 
print(f"✓ Mask predictions: {predM.shape}")
print(f"✓ Setup completed successfully!")
```

Save as `test_setup.py` and run:
```bash
python test_setup.py
```

## Configuration for Different Environments

### Memory-Constrained Systems

For systems with limited RAM (< 8GB):

```python
from redpan import create_memory_optimized_picker

picker = create_memory_optimized_picker(
    model=model,
    batch_size=4,           # Small batches
    pred_interval_sec=15.0  # Larger intervals
)
```

### High-Performance Systems

For systems with ample resources:

```python
from redpan import create_speed_optimized_picker

picker = create_speed_optimized_picker(
    model=model,
    batch_size=64,          # Large batches
    pred_interval_sec=5.0   # Smaller intervals for better resolution
)
```

### GPU Systems

For systems with CUDA-enabled GPUs:

```python
import tensorflow as tf

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Use large batch sizes for GPU efficiency
picker = create_picker(
    model=model,
    batch_size=128,  # Very large batches for GPU
    pred_interval_sec=5.0
)
```

## Integration with Existing RED-PAN Workflows

### Replacing Original RED-PAN Picker

**Before** (original RED-PAN):
```python
# Original RED-PAN setup
import sys
sys.path.append('./REDPAN_tools')
from REDPAN_picker import picker_info

# Complex initialization
picker = picker_info(
    mdl_hdr=mdl_hdr,
    model_h5=model_h5,
    pred_npts=pred_npts,
    dt=dt,
    pred_interval_sec=pred_interval_sec,
    postprocess_config=postprocess_config
)

# Processing loop with lists
pred_P, pred_S, pred_M = [], [], []
for window in sliding_windows:
    p, s, m = picker.process_window(window)
    pred_P.extend(p)
    pred_S.extend(s)
    pred_M.extend(m)

# Apply MedianFilter
final_predictions = apply_median_filter(pred_P, pred_S, pred_M)
```

**After** (REDPAN):
```python
# REDPAN setup (much simpler!)
from redpan import create_picker

# Load your existing model
model = tf.keras.models.load_model(model_h5)

# Create picker with same parameters
picker = create_picker(
    model=model,
    pred_npts=pred_npts,
    dt=dt,
    pred_interval_sec=pred_interval_sec,
    postprocess_config=postprocess_config
)

# Single function call for entire waveform
predP, predS, predM = picker.predict(waveform)
```

### Batch Processing Multiple Files

```python
import glob
from obspy import read
from redpan import create_picker

# Setup
model = tf.keras.models.load_model('path/to/model.h5')
picker = create_picker(model=model)

# Process multiple files
sac_files = glob.glob('data/*.SAC')
all_results = []

for sac_file in sac_files:
    # Load data
    st = read(sac_file)
    trace = st[0]
    
    # Ensure correct sampling rate
    if trace.stats.sampling_rate != 100:
        trace.resample(100)
    
    # Run picking
    predP, predS, predM = picker.predict(trace.data)
    
    # Store results
    results = {
        'file': sac_file,
        'starttime': trace.stats.starttime,
        'predP': predP,
        'predS': predS, 
        'predM': predM
    }
    all_results.append(results)
    
    print(f"Processed {sac_file}: {len(trace.data)} samples")

print(f"Completed processing {len(all_results)} files")
```

## Environment Variables

You can configure REDPAN behavior using environment variables:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Threading
export NUMEXPR_MAX_THREADS=16
export OMP_NUM_THREADS=8

# Memory management
export TF_GPU_MEMORY_FRACTION=0.8
```

Or in Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'redpan'
   ```
   **Solution**: Ensure the package is in your Python path or install in development mode.

2. **Memory Errors**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Reduce batch size or use `create_memory_optimized_picker()`.

3. **GPU Not Detected**
   ```
   No GPU devices available
   ```
   **Solution**: Check CUDA installation and TensorFlow GPU support.

4. **Slow Performance**
   ```
   Processing slower than expected
   ```
   **Solution**: Increase batch size, check GPU usage, ensure proper TensorFlow configuration.

### Performance Diagnostics

```python
import time
import psutil
from redpan import create_picker

def diagnose_performance():
    """Diagnose system performance for REDPAN"""
    
    print("REDPAN Performance Diagnostics")
    print("=" * 40)
    
    # System info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # GPU info
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    except:
        print("TensorFlow GPU check failed")
    
    # Quick benchmark
    model = DummyModel()  # Your dummy model from verification
    picker = create_picker(model=model, batch_size=16)
    
    test_waveform = np.random.normal(0, 0.1, 60000)  # 10 minutes
    
    start_time = time.time()
    predP, predS, predM = picker.predict(test_waveform)
    elapsed_time = time.time() - start_time
    
    real_time_factor = (len(test_waveform) * 0.01) / elapsed_time
    
    print(f"\nBenchmark Results:")
    print(f"Processing time: {elapsed_time:.3f} seconds")
    print(f"Real-time factor: {real_time_factor:.1f}x")
    
    if real_time_factor > 10:
        print("✓ Performance: Excellent")
    elif real_time_factor > 5:
        print("✓ Performance: Good")
    elif real_time_factor > 1:
        print("⚠ Performance: Acceptable")
    else:
        print("✗ Performance: Poor - check configuration")

# Run diagnostics
diagnose_performance()
```

### Getting Help

1. **Check the examples**: Run the provided example scripts
2. **Run diagnostics**: Use the performance diagnostic script above
3. **Check logs**: Enable detailed logging to see what's happening
4. **Test with dummy data**: Verify with synthetic data first
5. **Compare with original**: Benchmark against original RED-PAN if available

## Next Steps

After successful installation:

1. **Run examples**: Try the basic and advanced examples
2. **Test with your data**: Use your actual seismic data and models
3. **Optimize configuration**: Tune batch sizes and intervals for your system
4. **Integrate with existing workflows**: Replace original RED-PAN calls
5. **Monitor performance**: Use the benchmark suite to track improvements
