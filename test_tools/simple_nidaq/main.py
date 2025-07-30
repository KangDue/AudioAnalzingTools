#!/usr/bin/env python3
"""
NI DAQ Data Acquisition Application
A comprehensive tool for real-time data acquisition using NI DAQ devices
with configurable channels, live plotting, and automatic data saving.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel, QGroupBox, QFormLayout, QTextEdit,
    QMessageBox, QFileDialog, QGridLayout, QSplitter, QLineEdit
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

import pyqtgraph as pg
import nidaqmx
from nidaqmx.constants import (
    TerminalConfiguration, VoltageUnits, AccelUnits, SoundPressureUnits,
    CurrentUnits, ExcitationSource, AccelSensitivityUnits
)


@dataclass
class ChannelConfig:
    """Configuration for a single DAQ channel"""
    enabled: bool = False
    channel_type: str = "voltage"  # voltage, acceleration, microphone
    min_val: float = -10.0
    max_val: float = 10.0
    units: str = "Volts"
    terminal_config: str = "DEFAULT"  # DEFAULT, RSE, NRSE, DIFFERENTIAL, PSEUDODIFFERENTIAL
    custom_scale_name: str = ""
    
    # IEPE/Excitation parameters
    iepe_enabled: bool = False
    current_excit_source: str = "INTERNAL"  # INTERNAL, EXTERNAL, NONE
    current_excit_val: float = 0.004  # 4mA default
    voltage_excit_source: str = "INTERNAL"  # INTERNAL, EXTERNAL, NONE
    voltage_excit_val: float = 0.0  # Voltage excitation value
    use_excit_for_scaling: bool = False  # For ratiometric transducers
    
    # Sensor-specific parameters
    sensitivity: float = 1.0
    sensitivity_units: str = "MILLIVOLTS_PER_G"  # For accelerometers
    max_sound_pressure: float = 140.0  # dB SPL for microphones
    
    # Additional voltage channel parameters
    bridge_config: str = "NO_BRIDGE"  # NO_BRIDGE, FULL_BRIDGE, HALF_BRIDGE, QUARTER_BRIDGE
    
    # Legacy parameter names for backward compatibility
    @property
    def excitation_current(self) -> float:
        return self.current_excit_val
    
    @excitation_current.setter
    def excitation_current(self, value: float):
        self.current_excit_val = value
    
    @property
    def excitation_source(self) -> str:
        return self.current_excit_source
    
    @excitation_source.setter
    def excitation_source(self, value: str):
        self.current_excit_source = value


@dataclass
class MeasurementConfig:
    """Main measurement configuration"""
    sampling_rate: float = 1000.0
    samples_per_read: int = 1000
    plot_window_length: float = 10.0  # seconds
    auto_save_enabled: bool = False
    auto_save_interval: float = 60.0  # seconds
    save_directory: str = "./data"


class DataAcquisitionThread(QThread):
    """Thread for continuous data acquisition"""
    data_ready = pyqtSignal(np.ndarray, list)  # data, channel_names
    error_occurred = pyqtSignal(str)
    
    def __init__(self, device_name: str, channel_configs: Dict[str, ChannelConfig], 
                 measurement_config: MeasurementConfig):
        super().__init__()
        self.device_name = device_name
        self.channel_configs = channel_configs
        self.measurement_config = measurement_config
        self.running = False
        self.task = None
        
    def setup_task(self):
        """Setup the DAQ task with configured channels"""
        try:
            self.task = nidaqmx.Task()
            enabled_channels = []
            
            for ch_name, config in self.channel_configs.items():
                if not config.enabled:
                    continue
                    
                channel_name = f"{self.device_name}/{ch_name}"
                enabled_channels.append(ch_name)
                
                # Convert string constants to nidaqmx constants
                terminal_config_map = {
                    "DEFAULT": TerminalConfiguration.DEFAULT,
                    "RSE": TerminalConfiguration.RSE,
                    "NRSE": TerminalConfiguration.NRSE,
                    "DIFFERENTIAL": TerminalConfiguration.DIFFERENTIAL,
                    "PSEUDODIFFERENTIAL": TerminalConfiguration.PSEUDODIFFERENTIAL
                }
                
                excit_source_map = {
                    "INTERNAL": ExcitationSource.INTERNAL,
                    "EXTERNAL": ExcitationSource.EXTERNAL,
                    "NONE": ExcitationSource.NONE
                }
                
                sensitivity_units_map = {
                    "MILLIVOLTS_PER_G": AccelSensitivityUnits.MILLIVOLTS_PER_G,
                    "VOLTS_PER_G": AccelSensitivityUnits.VOLTS_PER_G
                }
                
                terminal_config = terminal_config_map.get(config.terminal_config, TerminalConfiguration.DEFAULT)
                current_excit_source = excit_source_map.get(config.current_excit_source, ExcitationSource.NONE)
                voltage_excit_source = excit_source_map.get(config.voltage_excit_source, ExcitationSource.NONE)
                
                if config.channel_type == "voltage":
                    self.task.ai_channels.add_ai_voltage_chan(
                        channel_name,
                        name_to_assign_to_channel="",
                        terminal_config=terminal_config,
                        min_val=config.min_val,
                        max_val=config.max_val,
                        units=VoltageUnits.VOLTS,
                        custom_scale_name=config.custom_scale_name
                    )
                elif config.channel_type == "acceleration":
                    sensitivity_units = sensitivity_units_map.get(config.sensitivity_units, AccelSensitivityUnits.MILLIVOLTS_PER_G)
                    
                    self.task.ai_channels.add_ai_accel_chan(
                        channel_name,
                        name_to_assign_to_channel="",
                        terminal_config=terminal_config,
                        min_val=config.min_val,
                        max_val=config.max_val,
                        units=AccelUnits.G,
                        sensitivity=config.sensitivity,
                        sensitivity_units=sensitivity_units,
                        current_excit_source=current_excit_source if config.iepe_enabled else ExcitationSource.NONE,
                        current_excit_val=config.current_excit_val if config.iepe_enabled else 0.0,
                        custom_scale_name=config.custom_scale_name
                    )
                elif config.channel_type == "microphone":
                    self.task.ai_channels.add_ai_microphone_chan(
                        channel_name,
                        name_to_assign_to_channel="",
                        terminal_config=terminal_config,
                        min_val=config.min_val,
                        max_val=config.max_val,
                        units=SoundPressureUnits.PA,
                        mic_sensitivity=config.sensitivity,
                        max_snd_press_level=config.max_sound_pressure,
                        current_excit_source=current_excit_source if config.iepe_enabled else ExcitationSource.NONE,
                        current_excit_val=config.current_excit_val if config.iepe_enabled else 0.0,
                        custom_scale_name=config.custom_scale_name
                    )
                elif config.channel_type == "acceleration_4wire":
                    sensitivity_units = sensitivity_units_map.get(config.sensitivity_units, AccelSensitivityUnits.MILLIVOLTS_PER_G)
                    
                    self.task.ai_channels.add_ai_accel_4_wire_dc_voltage_chan(
                        channel_name,
                        name_to_assign_to_channel="",
                        terminal_config=terminal_config,
                        min_val=config.min_val,
                        max_val=config.max_val,
                        units=AccelUnits.G,
                        sensitivity=config.sensitivity,
                        sensitivity_units=sensitivity_units,
                        voltage_excit_source=voltage_excit_source,
                        voltage_excit_val=config.voltage_excit_val,
                        use_excit_for_scaling=config.use_excit_for_scaling,
                        custom_scale_name=config.custom_scale_name
                    )
            
            if not enabled_channels:
                raise ValueError("No channels enabled for measurement")
                
            # Configure timing
            self.task.timing.cfg_samp_clk_timing(
                rate=self.measurement_config.sampling_rate,
                samps_per_chan=self.measurement_config.samples_per_read
            )
            
            return enabled_channels
            
        except Exception as e:
            self.error_occurred.emit(f"Task setup failed: {str(e)}")
            return None
    
    def run(self):
        """Main acquisition loop"""
        enabled_channels = self.setup_task()
        if not enabled_channels:
            return
            
        self.running = True
        
        try:
            self.task.start()
            
            while self.running:
                try:
                    data = self.task.read(
                        number_of_samples_per_channel=self.measurement_config.samples_per_read,
                        timeout=10.0
                    )
                    
                    # Convert to numpy array
                    if len(enabled_channels) == 1:
                        data = np.array([data])
                    else:
                        data = np.array(data)
                    
                    self.data_ready.emit(data, enabled_channels)
                    
                except nidaqmx.DaqError as e:
                    if self.running:  # Only emit error if we're still supposed to be running
                        self.error_occurred.emit(f"DAQ Error: {str(e)}")
                        break
                        
        except Exception as e:
            self.error_occurred.emit(f"Acquisition error: {str(e)}")
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
    
    def stop(self):
        """Stop the acquisition"""
        self.running = False
        self.wait()


class ChannelConfigWidget(QWidget):
    """Widget for configuring a single channel"""
    
    def __init__(self, channel_name: str, config: ChannelConfig):
        super().__init__()
        self.channel_name = channel_name
        self.config = config
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        layout = QFormLayout()
        
        # Enable checkbox
        self.enabled_cb = QCheckBox("Enable Channel")
        layout.addRow(self.enabled_cb)
        
        # Channel type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["voltage", "acceleration", "microphone", "acceleration_4wire"])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        layout.addRow("Channel Type:", self.type_combo)
        
        # Terminal configuration
        self.terminal_config_combo = QComboBox()
        self.terminal_config_combo.addItems(["DEFAULT", "RSE", "NRSE", "DIFFERENTIAL", "PSEUDODIFFERENTIAL"])
        layout.addRow("Terminal Config:", self.terminal_config_combo)
        
        # Custom scale name
        self.custom_scale_edit = QLineEdit()
        layout.addRow("Custom Scale Name:", self.custom_scale_edit)
        
        # Range
        self.min_val_spin = QDoubleSpinBox()
        self.min_val_spin.setRange(-1000, 1000)
        self.min_val_spin.setDecimals(3)
        layout.addRow("Min Value:", self.min_val_spin)
        
        self.max_val_spin = QDoubleSpinBox()
        self.max_val_spin.setRange(-1000, 1000)
        self.max_val_spin.setDecimals(3)
        layout.addRow("Max Value:", self.max_val_spin)
        
        # IEPE/Excitation settings
        iepe_group = QGroupBox("IEPE/Excitation Settings")
        iepe_layout = QFormLayout()
        
        self.iepe_enabled_cb = QCheckBox("Enable IEPE")
        iepe_layout.addRow(self.iepe_enabled_cb)
        
        # Current excitation
        self.current_excit_source_combo = QComboBox()
        self.current_excit_source_combo.addItems(["INTERNAL", "EXTERNAL", "NONE"])
        iepe_layout.addRow("Current Excit Source:", self.current_excit_source_combo)
        
        self.current_excit_val_spin = QDoubleSpinBox()
        self.current_excit_val_spin.setRange(0.0, 0.020)
        self.current_excit_val_spin.setDecimals(4)
        self.current_excit_val_spin.setSuffix(" A")
        iepe_layout.addRow("Current Excit Value:", self.current_excit_val_spin)
        
        # Voltage excitation
        self.voltage_excit_source_combo = QComboBox()
        self.voltage_excit_source_combo.addItems(["INTERNAL", "EXTERNAL", "NONE"])
        iepe_layout.addRow("Voltage Excit Source:", self.voltage_excit_source_combo)
        
        self.voltage_excit_val_spin = QDoubleSpinBox()
        self.voltage_excit_val_spin.setRange(0.0, 10.0)
        self.voltage_excit_val_spin.setDecimals(3)
        self.voltage_excit_val_spin.setSuffix(" V")
        iepe_layout.addRow("Voltage Excit Value:", self.voltage_excit_val_spin)
        
        # Use excitation for scaling
        self.use_excit_for_scaling_cb = QCheckBox("Use Excitation for Scaling")
        iepe_layout.addRow(self.use_excit_for_scaling_cb)
        
        iepe_group.setLayout(iepe_layout)
        layout.addRow(iepe_group)
        
        # Legacy excitation controls (for backward compatibility)
        self.excitation_current_spin = self.current_excit_val_spin  # Alias
        
        # Sensor-specific settings
        sensor_group = QGroupBox("Sensor Settings")
        sensor_layout = QFormLayout()
        
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.001, 1000)
        self.sensitivity_spin.setDecimals(6)
        sensor_layout.addRow("Sensitivity:", self.sensitivity_spin)
        
        # Sensitivity units (for accelerometers)
        self.sensitivity_units_combo = QComboBox()
        self.sensitivity_units_combo.addItems(["MILLIVOLTS_PER_G", "VOLTS_PER_G"])
        sensor_layout.addRow("Sensitivity Units:", self.sensitivity_units_combo)
        
        self.max_spl_spin = QDoubleSpinBox()
        self.max_spl_spin.setRange(80, 180)
        self.max_spl_spin.setSuffix(" dB SPL")
        sensor_layout.addRow("Max Sound Pressure:", self.max_spl_spin)
        
        # Bridge configuration (for voltage channels)
        self.bridge_config_combo = QComboBox()
        self.bridge_config_combo.addItems(["NO_BRIDGE", "FULL_BRIDGE", "HALF_BRIDGE", "QUARTER_BRIDGE"])
        sensor_layout.addRow("Bridge Config:", self.bridge_config_combo)
        
        sensor_group.setLayout(sensor_layout)
        layout.addRow(sensor_group)
        
        self.setLayout(layout)
        
    def on_type_changed(self, channel_type: str):
        """Update UI based on channel type"""
        # Show/hide relevant controls based on channel type
        iepe_relevant = channel_type in ["acceleration", "microphone"]
        voltage_excit_relevant = channel_type in ["acceleration_4wire"]
        mic_relevant = channel_type == "microphone"
        accel_relevant = channel_type in ["acceleration", "acceleration_4wire"]
        voltage_relevant = channel_type == "voltage"
        
        # IEPE/Current excitation controls
        self.iepe_enabled_cb.setVisible(iepe_relevant)
        self.current_excit_source_combo.setVisible(iepe_relevant or voltage_excit_relevant)
        self.current_excit_val_spin.setVisible(iepe_relevant or voltage_excit_relevant)
        
        # Voltage excitation controls
        self.voltage_excit_source_combo.setVisible(voltage_excit_relevant)
        self.voltage_excit_val_spin.setVisible(voltage_excit_relevant)
        self.use_excit_for_scaling_cb.setVisible(voltage_excit_relevant)
        
        # Sensor-specific controls
        self.max_spl_spin.setVisible(mic_relevant)
        self.sensitivity_units_combo.setVisible(accel_relevant)
        self.bridge_config_combo.setVisible(voltage_relevant)
        
        # Set default values based on type
        if channel_type == "voltage":
            self.min_val_spin.setValue(-10.0)
            self.max_val_spin.setValue(10.0)
            self.sensitivity_spin.setValue(1.0)
            self.terminal_config_combo.setCurrentText("DEFAULT")
            self.bridge_config_combo.setCurrentText("NO_BRIDGE")
        elif channel_type == "acceleration":
            self.min_val_spin.setValue(-50.0)
            self.max_val_spin.setValue(50.0)
            self.sensitivity_spin.setValue(100.0)  # 100 mV/g
            self.sensitivity_units_combo.setCurrentText("MILLIVOLTS_PER_G")
            self.iepe_enabled_cb.setChecked(True)
            self.current_excit_source_combo.setCurrentText("INTERNAL")
            self.current_excit_val_spin.setValue(0.004)
            self.terminal_config_combo.setCurrentText("DEFAULT")
        elif channel_type == "acceleration_4wire":
            self.min_val_spin.setValue(-50.0)
            self.max_val_spin.setValue(50.0)
            self.sensitivity_spin.setValue(100.0)  # 100 mV/g
            self.sensitivity_units_combo.setCurrentText("MILLIVOLTS_PER_G")
            self.voltage_excit_source_combo.setCurrentText("INTERNAL")
            self.voltage_excit_val_spin.setValue(5.0)
            self.use_excit_for_scaling_cb.setChecked(False)
            self.terminal_config_combo.setCurrentText("DEFAULT")
        elif channel_type == "microphone":
            self.min_val_spin.setValue(-1.0)
            self.max_val_spin.setValue(1.0)
            self.sensitivity_spin.setValue(50.0)  # 50 mV/Pa
            self.max_spl_spin.setValue(140.0)
            self.iepe_enabled_cb.setChecked(True)
            self.current_excit_source_combo.setCurrentText("INTERNAL")
            self.current_excit_val_spin.setValue(0.004)
            self.terminal_config_combo.setCurrentText("DEFAULT")
    
    def load_config(self):
        """Load configuration into UI"""
        self.enabled_cb.setChecked(self.config.enabled)
        self.type_combo.setCurrentText(self.config.channel_type)
        self.terminal_config_combo.setCurrentText(self.config.terminal_config)
        self.custom_scale_edit.setText(self.config.custom_scale_name)
        self.min_val_spin.setValue(self.config.min_val)
        self.max_val_spin.setValue(self.config.max_val)
        
        # IEPE/Excitation settings
        self.iepe_enabled_cb.setChecked(self.config.iepe_enabled)
        self.current_excit_source_combo.setCurrentText(self.config.current_excit_source)
        self.current_excit_val_spin.setValue(self.config.current_excit_val)
        self.voltage_excit_source_combo.setCurrentText(self.config.voltage_excit_source)
        self.voltage_excit_val_spin.setValue(self.config.voltage_excit_val)
        self.use_excit_for_scaling_cb.setChecked(self.config.use_excit_for_scaling)
        
        # Sensor settings
        self.sensitivity_spin.setValue(self.config.sensitivity)
        self.sensitivity_units_combo.setCurrentText(self.config.sensitivity_units)
        self.max_spl_spin.setValue(self.config.max_sound_pressure)
        self.bridge_config_combo.setCurrentText(self.config.bridge_config)
        
        self.on_type_changed(self.config.channel_type)
    
    def get_config(self) -> ChannelConfig:
        """Get current configuration from UI"""
        return ChannelConfig(
            enabled=self.enabled_cb.isChecked(),
            channel_type=self.type_combo.currentText(),
            min_val=self.min_val_spin.value(),
            max_val=self.max_val_spin.value(),
            terminal_config=self.terminal_config_combo.currentText(),
            custom_scale_name=self.custom_scale_edit.text(),
            
            # IEPE/Excitation parameters
            iepe_enabled=self.iepe_enabled_cb.isChecked(),
            current_excit_source=self.current_excit_source_combo.currentText(),
            current_excit_val=self.current_excit_val_spin.value(),
            voltage_excit_source=self.voltage_excit_source_combo.currentText(),
            voltage_excit_val=self.voltage_excit_val_spin.value(),
            use_excit_for_scaling=self.use_excit_for_scaling_cb.isChecked(),
            
            # Sensor parameters
            sensitivity=self.sensitivity_spin.value(),
            sensitivity_units=self.sensitivity_units_combo.currentText(),
            max_sound_pressure=self.max_spl_spin.value(),
            bridge_config=self.bridge_config_combo.currentText()
        )


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NI DAQ Data Acquisition Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize data storage
        self.data_buffer = {}
        self.time_buffer = []
        self.acquisition_thread = None
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_data)
        
        # Load settings
        self.settings_file = "daq_settings.json"
        self.load_settings()
        
        # Setup UI
        self.setup_ui()
        self.detect_devices()
        
    def setup_ui(self):
        """Setup the main user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        device_layout.addWidget(self.device_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.detect_devices)
        device_layout.addWidget(refresh_btn)
        device_layout.addStretch()
        main_layout.addLayout(device_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Configuration
        config_widget = QWidget()
        config_layout = QVBoxLayout()
        
        # Tabs for configuration
        self.config_tabs = QTabWidget()
        
        # Measurement config tab
        meas_tab = QWidget()
        meas_layout = QFormLayout()
        
        self.sampling_rate_spin = QDoubleSpinBox()
        self.sampling_rate_spin.setRange(1, 1000000)
        self.sampling_rate_spin.setValue(1000)
        self.sampling_rate_spin.setSuffix(" Hz")
        meas_layout.addRow("Sampling Rate:", self.sampling_rate_spin)
        
        self.samples_per_read_spin = QSpinBox()
        self.samples_per_read_spin.setRange(1, 100000)
        self.samples_per_read_spin.setValue(1000)
        meas_layout.addRow("Samples per Read:", self.samples_per_read_spin)
        
        self.plot_window_spin = QDoubleSpinBox()
        self.plot_window_spin.setRange(1, 3600)
        self.plot_window_spin.setValue(10)
        self.plot_window_spin.setSuffix(" s")
        meas_layout.addRow("Plot Window:", self.plot_window_spin)
        
        # Auto-save settings
        self.auto_save_cb = QCheckBox("Enable Auto-save")
        meas_layout.addRow(self.auto_save_cb)
        
        self.auto_save_interval_spin = QDoubleSpinBox()
        self.auto_save_interval_spin.setRange(1, 3600)
        self.auto_save_interval_spin.setValue(60)
        self.auto_save_interval_spin.setSuffix(" s")
        meas_layout.addRow("Auto-save Interval:", self.auto_save_interval_spin)
        
        meas_tab.setLayout(meas_layout)
        self.config_tabs.addTab(meas_tab, "Measurement")
        
        # Channel configuration tabs
        self.channel_widgets = {}
        for ch_name in ["ai0", "ai1", "ai2", "ai3"]:
            config = self.channel_configs.get(ch_name, ChannelConfig())
            widget = ChannelConfigWidget(ch_name, config)
            self.channel_widgets[ch_name] = widget
            self.config_tabs.addTab(widget, ch_name.upper())
        
        config_layout.addWidget(self.config_tabs)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Measurement")
        self.start_btn.clicked.connect(self.start_measurement)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Measurement")
        self.stop_btn.clicked.connect(self.stop_measurement)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self.save_data_manually)
        control_layout.addWidget(save_btn)
        
        config_layout.addLayout(control_layout)
        config_widget.setLayout(config_layout)
        
        # Right panel - Plotting and status
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        
        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Real-time Data")
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.addLegend()
        plot_layout.addWidget(self.plot_widget)
        
        # Status/log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        plot_layout.addWidget(QLabel("Status Log:"))
        plot_layout.addWidget(self.log_text)
        
        plot_widget.setLayout(plot_layout)
        
        # Add to splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(plot_widget)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        
        # Initialize plot curves
        self.plot_curves = {}
        colors = ['r', 'g', 'b', 'y']
        for i, ch_name in enumerate(["ai0", "ai1", "ai2", "ai3"]):
            curve = self.plot_widget.plot(pen=colors[i], name=ch_name.upper())
            self.plot_curves[ch_name] = curve
    
    def detect_devices(self):
        """Detect available NI DAQ devices"""
        try:
            system = nidaqmx.system.System.local()
            devices = [device.name for device in system.devices]
            
            self.device_combo.clear()
            if devices:
                self.device_combo.addItems(devices)
                self.log_message(f"Found {len(devices)} device(s): {', '.join(devices)}")
            else:
                self.log_message("No NI DAQ devices found")
                
        except Exception as e:
            self.log_message(f"Error detecting devices: {str(e)}")
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def start_measurement(self):
        """Start data acquisition"""
        if not self.device_combo.currentText():
            QMessageBox.warning(self, "Warning", "No device selected")
            return
        
        # Get current configuration
        device_name = self.device_combo.currentText()
        channel_configs = {name: widget.get_config() 
                          for name, widget in self.channel_widgets.items()}
        
        # Check if any channels are enabled
        if not any(config.enabled for config in channel_configs.values()):
            QMessageBox.warning(self, "Warning", "No channels enabled")
            return
        
        measurement_config = MeasurementConfig(
            sampling_rate=self.sampling_rate_spin.value(),
            samples_per_read=self.samples_per_read_spin.value(),
            plot_window_length=self.plot_window_spin.value(),
            auto_save_enabled=self.auto_save_cb.isChecked(),
            auto_save_interval=self.auto_save_interval_spin.value()
        )
        
        # Save current settings
        self.save_settings()
        
        # Initialize data buffers
        self.data_buffer = {name: [] for name, config in channel_configs.items() 
                           if config.enabled}
        self.time_buffer = []
        
        # Start acquisition thread
        self.acquisition_thread = DataAcquisitionThread(
            device_name, channel_configs, measurement_config
        )
        self.acquisition_thread.data_ready.connect(self.on_data_ready)
        self.acquisition_thread.error_occurred.connect(self.on_acquisition_error)
        self.acquisition_thread.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start auto-save timer if enabled
        if measurement_config.auto_save_enabled:
            self.auto_save_timer.start(int(measurement_config.auto_save_interval * 1000))
        
        self.log_message("Measurement started")
    
    def stop_measurement(self):
        """Stop data acquisition"""
        if self.acquisition_thread:
            self.acquisition_thread.stop()
            self.acquisition_thread = None
        
        self.auto_save_timer.stop()
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message("Measurement stopped")
    
    def on_data_ready(self, data: np.ndarray, channel_names: List[str]):
        """Handle new data from acquisition thread"""
        current_time = time.time()
        
        # Add data to buffers
        for i, ch_name in enumerate(channel_names):
            if ch_name in self.data_buffer:
                self.data_buffer[ch_name].extend(data[i])
        
        # Add time points
        dt = 1.0 / self.sampling_rate_spin.value()
        new_times = [current_time + j * dt for j in range(data.shape[1])]
        self.time_buffer.extend(new_times)
        
        # Limit buffer size based on plot window
        max_samples = int(self.plot_window_spin.value() * self.sampling_rate_spin.value())
        if len(self.time_buffer) > max_samples:
            excess = len(self.time_buffer) - max_samples
            self.time_buffer = self.time_buffer[excess:]
            for ch_name in self.data_buffer:
                self.data_buffer[ch_name] = self.data_buffer[ch_name][excess:]
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update real-time plots"""
        if not self.time_buffer:
            return
        
        # Convert to relative time (seconds from start)
        start_time = self.time_buffer[0]
        rel_times = [(t - start_time) for t in self.time_buffer]
        
        for ch_name, data in self.data_buffer.items():
            if ch_name in self.plot_curves and data:
                self.plot_curves[ch_name].setData(rel_times, data)
    
    def on_acquisition_error(self, error_message: str):
        """Handle acquisition errors"""
        self.log_message(f"Acquisition error: {error_message}")
        QMessageBox.critical(self, "Acquisition Error", error_message)
        self.stop_measurement()
    
    def auto_save_data(self):
        """Automatically save data at intervals"""
        if self.data_buffer and any(self.data_buffer.values()):
            self.save_data(auto_save=True)
    
    def save_data_manually(self):
        """Manually save current data"""
        if not self.data_buffer or not any(self.data_buffer.values()):
            QMessageBox.information(self, "Info", "No data to save")
            return
        
        self.save_data(auto_save=False)
    
    def save_data(self, auto_save: bool = False):
        """Save current data to file"""
        try:
            # Create data directory if it doesn't exist
            data_dir = "./data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            duration = len(self.time_buffer) / self.sampling_rate_spin.value() if self.time_buffer else 0
            filename = f"{timestamp}_duration_{duration:.1f}s.npy"
            filepath = os.path.join(data_dir, filename)
            
            # Prepare data for saving
            save_data = {
                'timestamp': timestamp,
                'sampling_rate': self.sampling_rate_spin.value(),
                'duration': duration,
                'channels': {},
                'time': self.time_buffer.copy() if self.time_buffer else [],
                'channel_configs': {name: asdict(widget.get_config()) 
                                  for name, widget in self.channel_widgets.items()}
            }
            
            for ch_name, data in self.data_buffer.items():
                save_data['channels'][ch_name] = data.copy()
            
            # Save to file
            np.save(filepath, save_data)
            
            prefix = "Auto-saved" if auto_save else "Saved"
            self.log_message(f"{prefix} data to {filename} ({len(self.time_buffer)} samples)")
            
        except Exception as e:
            error_msg = f"Failed to save data: {str(e)}"
            self.log_message(error_msg)
            if not auto_save:
                QMessageBox.critical(self, "Save Error", error_msg)
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                
                # Load channel configurations
                self.channel_configs = {}
                for ch_name in ["ai0", "ai1", "ai2", "ai3"]:
                    if ch_name in settings.get('channels', {}):
                        config_dict = settings['channels'][ch_name]
                        self.channel_configs[ch_name] = ChannelConfig(**config_dict)
                    else:
                        self.channel_configs[ch_name] = ChannelConfig()
                
                # Load measurement settings
                meas_settings = settings.get('measurement', {})
                self.measurement_config = MeasurementConfig(**meas_settings)
            else:
                # Default configurations
                self.channel_configs = {ch: ChannelConfig() for ch in ["ai0", "ai1", "ai2", "ai3"]}
                self.measurement_config = MeasurementConfig()
                
        except Exception as e:
            self.log_message(f"Failed to load settings: {str(e)}")
            self.channel_configs = {ch: ChannelConfig() for ch in ["ai0", "ai1", "ai2", "ai3"]}
            self.measurement_config = MeasurementConfig()
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            settings = {
                'channels': {name: asdict(widget.get_config()) 
                           for name, widget in self.channel_widgets.items()},
                'measurement': {
                    'sampling_rate': self.sampling_rate_spin.value(),
                    'samples_per_read': self.samples_per_read_spin.value(),
                    'plot_window_length': self.plot_window_spin.value(),
                    'auto_save_enabled': self.auto_save_cb.isChecked(),
                    'auto_save_interval': self.auto_save_interval_spin.value()
                }
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            self.log_message(f"Failed to save settings: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.acquisition_thread:
            self.stop_measurement()
        
        self.save_settings()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("NI DAQ Data Acquisition")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()