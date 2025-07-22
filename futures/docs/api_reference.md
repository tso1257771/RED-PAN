# REDPAN API Reference

## Core Classes

### REDPAN

Main class implementing SeisBench-style direct array accumulation for high-performance continuous seismic phase picking.

```python
class REDPAN:
    def __init__(self, model, pred_npts=6000, dt=0.01, pred_interval_sec=10.0, batch_size=32)
```

**Parameters:**
- `model`: TensorFlow model for prediction
- `pred_npts` (int): Model input length (default: 6000)
- `dt` (float): Sample rate (default: 0.01 for 100 Hz)
- `pred_interval_sec` (float): Sliding window step in seconds (default: 10.0)
- `batch_size` (int): Batch size for prediction (default: 32)

**Methods:**

#### predict(waveform)

Perform continuous phase picking on a waveform.

```python
def predict(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Parameters:**
- `waveform` (np.ndarray): Input waveform array

**Returns:**
- `predP` (np.ndarray): P phase predictions
- `predS` (np.ndarray): S phase predictions  
- `predM` (np.ndarray): Detection mask predictions

**Example:**
```python
from redpan import create_picker

picker = create_picker(model=my_model)
predP, predS, predM = picker.predict(waveform)
```

## Factory Functions

### create_picker

Recommended factory function for creating REDPAN instances.

```python
def create_picker(
    model,
    pred_npts: int = 6000,
    dt: float = 0.01,
    pred_interval_sec: float = 10.0,
    batch_size: int = 32,
    postprocess_config: Optional[Dict[str, Any]] = None
) -> REDPAN
```

**Parameters:**
- `model`: TensorFlow model for prediction
- `pred_npts` (int): Model input length (default: 6000)
- `dt` (float): Sample rate (default: 0.01 for 100 Hz)
- `pred_interval_sec` (float): Sliding window step (default: 10.0)
- `batch_size` (int): Batch size for prediction (default: 32)
- `postprocess_config` (dict, optional): Postprocessing configuration

**Example:**
```python
picker = create_picker(
    model=my_model,
    batch_size=64,  # Increase for powerful GPUs
    postprocess_config={'trigger_thre': 0.3}
)
```

### create_memory_optimized_picker

Factory for memory-constrained environments.

```python
def create_memory_optimized_picker(model, **kwargs) -> REDPAN
```

**Features:**
- Small batch size (4) for minimal memory usage
- Conservative prediction intervals
- Optimized for systems with limited RAM

### create_speed_optimized_picker

Factory for maximum processing speed.

```python
def create_speed_optimized_picker(model, **kwargs) -> REDPAN
```

**Features:**
- Large batch size (64) for GPU efficiency
- Aggressive prediction intervals
- Optimized for powerful systems

### create_compatibility_picker

Factory for maximum compatibility with original RED-PAN.

```python
def create_compatibility_picker(model, **kwargs) -> REDPAN
```

**Features:**
- Identical parameter defaults to original RED-PAN
- Conservative settings for compatibility
- Drop-in replacement

## Utility Functions

### create_gaussian_weights

Create Gaussian position weights for window blending.

```python
def create_gaussian_weights(length: int, sigma_fraction: float = 0.25) -> np.ndarray
```

**Parameters:**
- `length` (int): Length of weight array
- `sigma_fraction` (float): Gaussian width as fraction of length (default: 0.25)

**Returns:**
- `weights` (np.ndarray): Gaussian weight array with maximum value of 1.0

### validate_waveform

Validate input waveform for processing.

```python
def validate_waveform(
    waveform: np.ndarray, 
    expected_length: Optional[int] = None
) -> Tuple[bool, str]
```

**Parameters:**
- `waveform` (np.ndarray): Input waveform to validate
- `expected_length` (int, optional): Expected length for validation

**Returns:**
- `is_valid` (bool): Whether waveform is valid
- `message` (str): Validation message

### estimate_memory_usage

Estimate memory usage for processing.

```python
def estimate_memory_usage(
    waveform_length: int,
    pred_npts: int = 6000,
    batch_size: int = 32
) -> Dict[str, float]
```

**Parameters:**
- `waveform_length` (int): Length of input waveform
- `pred_npts` (int): Model input length
- `batch_size` (int): Batch size

**Returns:**
- Dictionary with memory estimates in MB

## Configuration

### Postprocessing Configuration

The `postprocess_config` parameter accepts a dictionary with the following options:

```python
postprocess_config = {
    'mask_trigger': [0.1, 0.1],      # Trigger thresholds
    'mask_len_thre': 0.5,            # Length threshold  
    'mask_err_win': 0.5,             # Error window
    'trigger_thre': 0.3              # Detection threshold
}
```

### Performance Tuning

#### Batch Size Selection

- **Small systems (< 8GB RAM)**: `batch_size=4-8`
- **Medium systems (8-16GB RAM)**: `batch_size=16-32`
- **Large systems (> 16GB RAM)**: `batch_size=32-64`
- **GPU systems**: `batch_size=64-128`

#### Prediction Interval

- **High resolution**: `pred_interval_sec=5.0` (more overlap, slower)
- **Standard**: `pred_interval_sec=10.0` (balanced)
- **Fast processing**: `pred_interval_sec=15.0` (less overlap, faster)

#### Memory vs Speed Trade-offs

```python
# Memory optimized (slower)
picker = create_memory_optimized_picker(
    model=model,
    batch_size=4,
    pred_interval_sec=15.0
)

# Speed optimized (more memory)
picker = create_speed_optimized_picker(
    model=model, 
    batch_size=64,
    pred_interval_sec=5.0
)
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values
- `AssertionError`: Failed validation checks
- `MemoryError`: Insufficient memory for processing
- `RuntimeError`: Model prediction failures

### Best Practices

1. Always validate inputs with `validate_waveform()`
2. Use factory functions rather than direct instantiation
3. Monitor memory usage for long waveforms
4. Choose appropriate batch sizes for your system
5. Handle edge cases (short waveforms, missing data)

## Migration from Original RED-PAN

### Before (Original RED-PAN)
```python
from REDPAN_picker import picker_info

# Complex setup required
pred_P, pred_S, pred_M = [], [], []
for window in sliding_windows:
    p, s, m = model.predict(window)
    pred_P.append(p)
    pred_S.append(s) 
    pred_M.append(m)

# Apply MedianFilter (slow)
final_P = apply_median_filter(np.concatenate(pred_P))
final_S = apply_median_filter(np.concatenate(pred_S))
final_M = apply_median_filter(np.concatenate(pred_M))
```

### After (REDPAN)
```python
from redpan import create_picker

# Simple, fast, one-line solution
picker = create_picker(model=model)
predP, predS, predM = picker.predict(waveform)
```

## Performance Characteristics

### Typical Performance

- **Processing speed**: 10-50x faster than original RED-PAN
- **Memory usage**: 2-5x lower than original approach
- **Real-time factor**: 50-200x (processes 50-200 seconds of data per second)
- **Accuracy**: Equal or better due to Gaussian position weights

### Scaling

- **Linear scaling** with waveform length
- **Efficient memory usage** regardless of input size
- **GPU acceleration** through optimized batching
- **Parallel processing** ready architecture
