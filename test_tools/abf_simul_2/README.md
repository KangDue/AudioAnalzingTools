# Ultra-Fast Audio Beam Focusing Simulator

A high-performance Python-based audio beam focusing simulator designed for spatial sound analysis using circular microphone arrays. This simulator implements efficient fractional delay filtering and FFT-based convolution to achieve real-time or near real-time performance.

## Features

### Core Capabilities
- **Circular Microphone Array Simulation**: Configurable array with 8-64 microphones
- **Beam Focusing (Not Beamforming)**: Implements delay-and-sum beam focusing for spatial audio analysis
- **FFT-Based Convolution**: Optimized frequency-domain processing for maximum speed
- **Fractional Delay Filters**: Sub-sample accurate delay computation using Lagrange interpolation
- **Real-Time Performance**: Optimized for real-time or near real-time processing
- **Energy Map Visualization**: Dynamic 2D energy maps with time evolution

### Technical Specifications
- **Default Configuration**: 32 microphones, 0.2m radius circular array
- **Target Plane**: 0.4m × 0.4m square at 0.4m distance, 70×70 grid resolution
- **Sample Rate**: 51,200 Hz (configurable)
- **Processing**: Vectorized NumPy operations with optional GPU acceleration
- **Output**: Time-evolving energy maps for source localization

## Installation

### Prerequisites
- Python 3.7 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

### Quick Install
```bash
# Clone or download the project
cd abf_simul_2

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies
For enhanced performance and features:
```bash
# GPU acceleration (optional)
pip install cupy>=10.0.0
pip install numba>=0.56.0

# Audio file support (optional)
pip install soundfile>=0.10.0
pip install librosa>=0.9.0

# Jupyter notebook support (optional)
pip install jupyter>=1.0.0
pip install ipywidgets>=7.6.0
```

## Quick Start

### Basic Usage
```python
from beam_focusing_simulator import BeamFocusingSimulator

# Create simulator with default parameters
simulator = BeamFocusingSimulator()

# Generate test audio (single source at center)
audio_data = simulator.generate_test_audio(duration=3.0)

# Compute beam focusing
energy_maps, time_stamps = simulator.compute_beam_focus(audio_data)

# Visualize results
anim = simulator.visualize_energy_maps(energy_maps, time_stamps)
```

### Custom Configuration
```python
# Create simulator with custom parameters
simulator = BeamFocusingSimulator(
    n_mics=64,              # 64 microphones
    array_radius=0.3,       # 0.3m radius
    target_distance=0.5,    # 0.5m target distance
    grid_resolution=100,    # 100×100 grid
    sample_rate=48000       # 48kHz sample rate
)

# Multiple source scenario
source_positions = [
    (-0.1, -0.1, 0.5),  # Source 1
    (0.1, 0.1, 0.5),   # Source 2
    (0.0, 0.0, 0.5)    # Source 3
]

audio_data = simulator.generate_test_audio(
    duration=5.0, 
    source_positions=source_positions
)

# High-resolution temporal analysis
energy_maps, time_stamps = simulator.compute_beam_focus(
    audio_data, 
    time_window=0.1,    # 100ms windows
    overlap=0.8         # 80% overlap
)
```

## Usage Examples

### 1. Run Demo Script
```bash
python demo.py
```
This runs a comprehensive demonstration including:
- Basic beam focusing
- Multiple source scenarios
- Parameter sensitivity analysis
- Performance benchmarking
- Advanced visualizations

### 2. Interactive Jupyter Notebook
```bash
jupyter notebook interactive_demo.ipynb
```
Provides interactive widgets for:
- Real-time parameter adjustment
- Live visualization updates
- Performance analysis
- Data export capabilities

### 3. Performance Benchmarking
```python
# Quick performance test
results = simulator.benchmark_performance(duration=2.0)
print(f"Real-time factor: {results['real_time_factor']:.2f}x")
```

## API Reference

### BeamFocusingSimulator Class

#### Constructor
```python
BeamFocusingSimulator(
    n_mics=32,              # Number of microphones
    array_radius=0.2,       # Array radius (m)
    target_distance=0.4,    # Target plane distance (m)
    target_size=0.4,        # Target plane size (m)
    grid_resolution=70,     # Grid points per dimension
    sample_rate=51200,      # Sample rate (Hz)
    sound_speed=343.0       # Speed of sound (m/s)
)
```

#### Key Methods

**generate_test_audio(duration, source_positions)**
- Generates multi-channel test audio with specified source positions
- Returns: `numpy.ndarray` (n_samples, n_mics)

**compute_beam_focus(audio_data, time_window, overlap)**
- Computes beam focusing energy maps over time
- Returns: `(energy_maps, time_stamps)` tuple

**visualize_energy_maps(energy_maps, time_stamps, save_animation)**
- Creates animated visualization of energy maps
- Returns: `matplotlib.animation.FuncAnimation`

**benchmark_performance(duration)**
- Runs performance benchmark
- Returns: Dictionary with timing and performance metrics

## Performance Optimization

### Computational Efficiency
The simulator is optimized for speed through:

1. **FFT-Based Convolution**: All delay filtering performed in frequency domain
2. **Vectorized Operations**: Extensive use of NumPy broadcasting
3. **Memory Optimization**: In-place operations where possible
4. **Batch Processing**: Grid points processed in batches

### Performance Targets
- **Real-time Factor**: >1.0x for typical configurations
- **Latency**: <100ms for 70×70 grid with 32 microphones
- **Throughput**: >1M grid points/second on modern hardware

### Scaling Guidelines
| Configuration | Expected RT Factor | Use Case |
|---------------|-------------------|----------|
| 16 mics, 50×50 grid | 5-10x | Fast prototyping |
| 32 mics, 70×70 grid | 2-5x | Standard analysis |
| 64 mics, 100×100 grid | 1-2x | High-resolution |

## Algorithm Details

### Beam Focusing Implementation
The simulator implements **beam focusing** (not traditional beamforming):

1. **Geometric Delay Calculation**: Precise time-of-flight from each grid point to each microphone
2. **Fractional Delay Filtering**: Lagrange interpolation for sub-sample delays
3. **Delay-and-Sum**: Coherent addition of delayed signals
4. **Energy Computation**: RMS energy calculation for each grid point

### Mathematical Foundation
For a target point **p** and microphone **m**:

```
delay_m = ||p - m|| / c
filtered_signal_m = fractional_delay(audio_m, delay_m)
focused_energy = RMS(Σ filtered_signal_m)
```

Where:
- `||p - m||` is the Euclidean distance
- `c` is the speed of sound
- `fractional_delay()` applies sub-sample delay
- `RMS()` computes root-mean-square energy

## File Structure

```
abf_simul_2/
├── beam_focusing_simulator.py    # Main simulator class
├── demo.py                       # Comprehensive demo script
├── interactive_demo.ipynb        # Jupyter notebook interface
├── requirements.txt              # Python dependencies
├── work_log.txt                 # Development log
├── README.md                    # This file
└── project_request.md           # Original project specification
```

## Troubleshooting

### Common Issues

**Slow Performance**
- Reduce grid resolution or number of microphones
- Increase time window size
- Check for memory constraints

**Memory Errors**
- Reduce audio duration or grid resolution
- Process in smaller chunks
- Close other applications

**Visualization Issues**
- Update matplotlib: `pip install --upgrade matplotlib`
- For Jupyter: `pip install ipywidgets`
- Enable widget extensions: `jupyter nbextension enable --py widgetsnbextension`

### Performance Tips

1. **Optimal Grid Size**: 70×70 provides good balance of resolution and speed
2. **Time Window**: 0.1-0.2s windows work well for most applications
3. **Overlap**: 50-80% overlap provides smooth temporal evolution
4. **Memory**: Monitor RAM usage for large configurations

## Contributing

Contributions are welcome! Areas for improvement:

- GPU acceleration with CuPy/CUDA
- Advanced beamforming algorithms (MVDR, MUSIC)
- Real-time audio input support
- 3D visualization capabilities
- Performance optimizations

## License

This project is provided as-is for research and educational purposes.

## References

The implementation is based on established acoustic beamforming principles:

1. Fractional delay filters and frequency-domain convolution
2. Circular microphone array acoustics
3. Spatial audio processing techniques
4. Real-time signal processing optimization

For detailed algorithmic references, see the original project specification.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the demo scripts and notebook examples
3. Examine the work log for development notes
4. Test with smaller configurations first

---

**Note**: This simulator implements beam **focusing** rather than traditional beamforming. The distinction is important: beam focusing computes energy at specific spatial points, while beamforming typically enhances signals from specific directions.