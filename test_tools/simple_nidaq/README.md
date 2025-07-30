# NI DAQ Data Acquisition Tool

A comprehensive Python application for real-time data acquisition using National Instruments DAQ devices. This tool provides an intuitive GUI for configuring channels, visualizing data in real-time, and automatically saving measurements.

## Features

### 🔧 Device Management
- Automatic detection of connected NI DAQ devices
- Device selection from dropdown list
- Real-time device status monitoring

### 📊 Channel Configuration
- Support for 4 analog input channels (ai0, ai1, ai2, ai3)
- Multiple channel types:
  - **Voltage**: Standard voltage measurements
  - **Acceleration**: Accelerometer sensors with IEPE support
  - **Microphone**: Microphone sensors with IEPE support
- Configurable parameters per channel:
  - Input range (min/max values)
  - IEPE excitation current (for accelerometers and microphones)
  - Sensor sensitivity
  - Maximum sound pressure level (for microphones)

### 📈 Real-time Visualization
- Live plotting using PyQtGraph for high performance
- Configurable plot window length (1-3600 seconds)
- Interactive zoom and pan capabilities
- Multi-channel display with color-coded traces
- Legend for easy channel identification

### 💾 Data Management
- Automatic data saving at configurable intervals
- Manual save functionality
- Data stored in NumPy format (.npy) with metadata
- Filename format: `YYYYMMDD_HHMMSS_duration_Xs.npy`
- Includes channel configurations and measurement parameters

### ⚙️ Measurement Control
- Configurable sampling rate (1 Hz to 1 MHz)
- Adjustable samples per read for performance optimization
- Start/stop measurement controls
- Real-time status logging

### 🔄 Settings Persistence
- Automatic saving and loading of all configurations
- Settings stored in `daq_settings.json`
- Preserves channel configurations between sessions

## Installation

### Prerequisites
- Python 3.7 or higher
- NI-DAQmx drivers installed on your system
- Compatible National Instruments DAQ device

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install nidaqmx>=0.6.5 PyQt5>=5.15.0 pyqtgraph>=0.13.0 numpy>=1.21.0 scipy>=1.7.0
```

## Usage

### Starting the Application

```bash
python main.py
```

### Basic Workflow

1. **Device Selection**
   - The application automatically detects connected DAQ devices
   - Select your device from the dropdown menu
   - Click "Refresh" if your device doesn't appear

2. **Channel Configuration**
   - Navigate to individual channel tabs (AI0, AI1, AI2, AI3)
   - Enable desired channels using the checkbox
   - Select appropriate channel type (voltage/acceleration/microphone)
   - Configure parameters based on your sensors:
     - **Voltage channels**: Set input range
     - **Accelerometers**: Enable IEPE, set sensitivity and excitation current
     - **Microphones**: Enable IEPE, set sensitivity, excitation current, and max SPL

3. **Measurement Setup**
   - Go to the "Measurement" tab
   - Set sampling rate appropriate for your application
   - Configure samples per read (affects update rate)
   - Set plot window length for visualization
   - Enable auto-save if desired and set interval

4. **Data Acquisition**
   - Click "Start Measurement" to begin
   - Monitor real-time data in the plot window
   - Use mouse to zoom and pan in the plot
   - Check status log for any messages or errors
   - Click "Stop Measurement" when done

5. **Data Management**
   - Data is automatically saved if auto-save is enabled
   - Use "Save Data" button for manual saves
   - Files are saved in the `./data` directory
   - Each file contains complete measurement data and metadata

### Configuration Tips

#### For Accelerometers
- Enable IEPE excitation
- Set excitation current to 4 mA (typical)
- Configure sensitivity in V/g (e.g., 0.1 for 100 mV/g sensors)
- Set appropriate input range based on expected acceleration levels

#### For Microphones
- Enable IEPE excitation
- Set excitation current according to microphone specifications
- Configure sensitivity in V/Pa (e.g., 0.05 for 50 mV/Pa)
- Set maximum sound pressure level (typically 140 dB SPL)

#### For Voltage Measurements
- Set input range to match expected signal levels
- No IEPE configuration needed
- Sensitivity typically set to 1.0

### Performance Optimization

- **High-speed acquisition**: Increase "Samples per Read" to reduce overhead
- **Real-time display**: Reduce plot window length for faster updates
- **Memory usage**: Enable auto-save to prevent excessive memory consumption
- **CPU usage**: Lower sampling rate if real-time performance is poor

## File Structure

```
simple_nidaq/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── work_log.txt        # Development log
├── daq_settings.json   # Saved configurations (created automatically)
└── data/               # Saved measurement files (created automatically)
    └── YYYYMMDD_HHMMSS_duration_Xs.npy
```

## Data Format

Saved data files (.npy) contain a dictionary with the following structure:

```python
{
    'timestamp': '20241219_143022',
    'sampling_rate': 1000.0,
    'duration': 60.5,
    'channels': {
        'ai0': [data_array],
        'ai1': [data_array],
        # ... other enabled channels
    },
    'time': [time_array],
    'channel_configs': {
        'ai0': {channel_configuration},
        # ... configurations for all channels
    }
}
```

### Loading Saved Data

```python
import numpy as np

# Load data file
data = np.load('data/20241219_143022_duration_60.5s.npy', allow_pickle=True).item()

# Access measurement data
ai0_data = data['channels']['ai0']
time_data = data['time']
sampling_rate = data['sampling_rate']

# Access channel configurations
ai0_config = data['channel_configs']['ai0']
```

## Troubleshooting

### Common Issues

1. **"No NI DAQ devices found"**
   - Ensure NI-DAQmx drivers are installed
   - Check device connections
   - Verify device is recognized by NI MAX (Measurement & Automation Explorer)

2. **"Task setup failed"**
   - Check channel configurations
   - Ensure IEPE settings match your sensors
   - Verify input ranges are appropriate

3. **"DAQ Error" during measurement**
   - Device may have been disconnected
   - Check for conflicting applications using the device
   - Restart the application

4. **Poor real-time performance**
   - Reduce sampling rate
   - Increase "Samples per Read"
   - Reduce plot window length
   - Close other applications

5. **File save errors**
   - Check disk space
   - Ensure write permissions in the data directory
   - Verify the data directory exists

### Error Messages

The application provides detailed error messages in the status log. Common error types:

- **Configuration errors**: Invalid parameter combinations
- **Hardware errors**: Device communication issues
- **File I/O errors**: Problems saving data or settings

## Technical Details

### Architecture
- **GUI Framework**: PyQt5 for cross-platform compatibility
- **Plotting**: PyQtGraph for high-performance real-time visualization
- **DAQ Interface**: nidaqmx Python library
- **Threading**: Separate acquisition thread to prevent GUI blocking
- **Data Storage**: NumPy arrays for efficient numerical data handling

### Performance Characteristics
- **Maximum sampling rate**: Limited by DAQ hardware and system performance
- **Channel count**: Fixed at 4 analog inputs (ai0-ai3)
- **Memory usage**: Scales with plot window length and sampling rate
- **Update rate**: Configurable via "Samples per Read" parameter

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with National Instruments software licensing terms when using NI-DAQmx drivers.

## Support

For issues related to:
- **NI hardware**: Consult National Instruments documentation and support
- **Python dependencies**: Check respective package documentation
- **Application bugs**: Review the status log for detailed error information

---

**Note**: This application requires compatible National Instruments DAQ hardware and properly installed NI-DAQmx drivers. Test with your specific hardware configuration before production use.