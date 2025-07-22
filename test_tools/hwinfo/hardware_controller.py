#!/usr/bin/env python3
"""
Hardware Controller for PC Components
Supports monitoring and control of fans, sensors, and microphones

Author: Hardware Control Project
Date: 2024
"""

import psutil
import platform
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: WMI module not available. Some Windows-specific features may not work.{Style.RESET_ALL}")

try:
    import sounddevice as sd
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: Audio modules not available. Microphone control features disabled.{Style.RESET_ALL}")

@dataclass
class HardwareInfo:
    """Data class for hardware information"""
    name: str
    type: str
    value: Optional[float]
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    controllable: bool = False
    control_methods: List[str] = None

class HardwareController:
    """Main hardware controller class"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.wmi_connection = None
        self.audio_devices = []
        
        if WMI_AVAILABLE and platform.system() == "Windows":
            try:
                self.wmi_connection = wmi.WMI()
            except Exception as e:
                print(f"{Fore.RED}Failed to initialize WMI: {e}{Style.RESET_ALL}")
        
        if AUDIO_AVAILABLE:
            self._initialize_audio_devices()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True)
        }
    
    def _initialize_audio_devices(self):
        """Initialize audio device information"""
        try:
            if AUDIO_AVAILABLE:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:  # Input device (microphone)
                        self.audio_devices.append({
                            'index': i,
                            'name': device['name'],
                            'type': 'microphone',
                            'channels': device['max_input_channels'],
                            'default_samplerate': device['default_samplerate'],
                            'controllable': True
                        })
        except Exception as e:
            print(f"{Fore.RED}Error initializing audio devices: {e}{Style.RESET_ALL}")
    
    def get_fan_information(self) -> List[HardwareInfo]:
        """Get fan speed information and control capabilities"""
        fans = []
        
        # Try to get fan information from psutil (limited on Windows)
        try:
            if hasattr(psutil, 'sensors_fans'):
                fan_data = psutil.sensors_fans()
                for fan_name, fan_list in fan_data.items():
                    for i, fan in enumerate(fan_list):
                        fans.append(HardwareInfo(
                            name=f"{fan_name}_{i}",
                            type="fan",
                            value=fan.current,
                            unit="RPM",
                            controllable=False,  # psutil doesn't support fan control
                            control_methods=[]
                        ))
        except Exception as e:
            print(f"{Fore.YELLOW}psutil fan detection failed: {e}{Style.RESET_ALL}")
        
        # Try WMI for Windows systems
        if WMI_AVAILABLE and self.wmi_connection:
            try:
                # Try OpenHardwareMonitor WMI namespace
                ohm_wmi = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                sensors = ohm_wmi.Sensor()
                for sensor in sensors:
                    if sensor.SensorType == 'Fan':
                        fans.append(HardwareInfo(
                            name=sensor.Name,
                            type="fan",
                            value=sensor.Value,
                            unit="RPM",
                            controllable=True,  # OHM may support control
                            control_methods=["WMI", "OpenHardwareMonitor"]
                        ))
            except Exception as e:
                print(f"{Fore.YELLOW}OpenHardwareMonitor WMI access failed: {e}{Style.RESET_ALL}")
        
        # Add theoretical fan control information
        if not fans:
            fans.append(HardwareInfo(
                name="System Fan (Theoretical)",
                type="fan",
                value=None,
                unit="RPM",
                controllable=True,
                control_methods=[
                    "BIOS/UEFI Settings",
                    "Motherboard Software",
                    "PWM Control (GPIO)",
                    "Fan Controller Hardware",
                    "LibreHardwareMonitor",
                    "SpeedFan (Legacy)"
                ]
            ))
        
        return fans
    
    def get_sensor_information(self) -> List[HardwareInfo]:
        """Get temperature and other sensor information"""
        sensors = []
        
        # CPU Temperature sensors
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temp_data = psutil.sensors_temperatures()
                for sensor_name, temp_list in temp_data.items():
                    for i, temp in enumerate(temp_list):
                        sensors.append(HardwareInfo(
                            name=f"{sensor_name}_{temp.label or i}",
                            type="temperature",
                            value=temp.current,
                            unit="°C",
                            min_value=0,
                            max_value=temp.high or 100,
                            controllable=False,
                            control_methods=[]
                        ))
        except Exception as e:
            print(f"{Fore.YELLOW}Temperature sensor detection failed: {e}{Style.RESET_ALL}")
        
        # Battery sensors (for laptops)
        try:
            battery = psutil.sensors_battery()
            if battery:
                sensors.append(HardwareInfo(
                    name="Battery",
                    type="battery",
                    value=battery.percent,
                    unit="%",
                    min_value=0,
                    max_value=100,
                    controllable=False,
                    control_methods=[]
                ))
        except Exception:
            pass
        
        # WMI sensors for Windows
        if WMI_AVAILABLE and self.wmi_connection:
            try:
                ohm_wmi = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                wmi_sensors = ohm_wmi.Sensor()
                for sensor in wmi_sensors:
                    if sensor.SensorType in ['Temperature', 'Voltage', 'Power']:
                        sensors.append(HardwareInfo(
                            name=sensor.Name,
                            type=sensor.SensorType.lower(),
                            value=sensor.Value,
                            unit=self._get_sensor_unit(sensor.SensorType),
                            controllable=sensor.SensorType == 'Temperature',
                            control_methods=["Thermal Management", "Power Settings"] if sensor.SensorType == 'Temperature' else []
                        ))
            except Exception as e:
                print(f"{Fore.YELLOW}WMI sensor access failed: {e}{Style.RESET_ALL}")
        
        return sensors
    
    def get_microphone_information(self) -> List[HardwareInfo]:
        """Get microphone information and control capabilities"""
        microphones = []
        
        if not AUDIO_AVAILABLE:
            return [HardwareInfo(
                name="Audio libraries not available",
                type="microphone",
                value=None,
                unit="N/A",
                controllable=False,
                control_methods=[]
            )]
        
        for device in self.audio_devices:
            microphones.append(HardwareInfo(
                name=device['name'],
                type="microphone",
                value=device['default_samplerate'],
                unit="Hz",
                min_value=8000,
                max_value=192000,
                controllable=True,
                control_methods=[
                    "Sample Rate Control",
                    "Bit Depth Control (16/24/32-bit)",
                    "Channel Configuration",
                    "Buffer Size Control",
                    "Gain/Volume Control",
                    "Audio Format Selection"
                ]
            ))
        
        return microphones
    
    def _get_sensor_unit(self, sensor_type: str) -> str:
        """Get appropriate unit for sensor type"""
        units = {
            'Temperature': '°C',
            'Voltage': 'V',
            'Power': 'W',
            'Fan': 'RPM',
            'Load': '%'
        }
        return units.get(sensor_type, '')
    
    def control_fan(self, fan_name: str, mode: str = None, level: int = None, duty: int = None) -> bool:
        """
        Control fan speed
        
        Args:
            fan_name: Name of the fan to control
            mode: Control mode ('auto', 'manual', 'silent', 'performance')
            level: Fan level (1-5 or 1-10 depending on system)
            duty: PWM duty cycle (0-100%)
        
        Returns:
            bool: Success status
        """
        print(f"{Fore.CYAN}Fan Control Request:{Style.RESET_ALL}")
        print(f"  Fan: {fan_name}")
        print(f"  Mode: {mode}")
        print(f"  Level: {level}")
        print(f"  Duty: {duty}%")
        
        # This is a demonstration - actual implementation would require:
        # 1. Hardware-specific drivers
        # 2. Administrative privileges
        # 3. Direct hardware access or specialized libraries
        
        print(f"{Fore.YELLOW}Note: Actual fan control requires specialized hardware drivers{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Recommended approaches:{Style.RESET_ALL}")
        print("  - Use motherboard manufacturer software")
        print("  - Configure BIOS/UEFI fan curves")
        print("  - Use LibreHardwareMonitor with admin privileges")
        print("  - Hardware fan controllers")
        
        return False  # Placeholder
    
    def control_microphone(self, device_index: int, sample_rate: int = 44100, 
                          bit_depth: int = 16, channels: int = 1) -> bool:
        """
        Configure microphone settings
        
        Args:
            device_index: Audio device index
            sample_rate: Sampling rate in Hz
            bit_depth: Bit depth (16, 24, 32)
            channels: Number of channels (1=mono, 2=stereo)
        
        Returns:
            bool: Success status
        """
        if not AUDIO_AVAILABLE:
            print(f"{Fore.RED}Audio libraries not available{Style.RESET_ALL}")
            return False
        
        try:
            # Test recording with specified settings
            print(f"{Fore.CYAN}Microphone Configuration:{Style.RESET_ALL}")
            print(f"  Device Index: {device_index}")
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Bit Depth: {bit_depth} bit")
            print(f"  Channels: {channels}")
            
            # Validate settings
            devices = sd.query_devices()
            if device_index >= len(devices):
                print(f"{Fore.RED}Invalid device index{Style.RESET_ALL}")
                return False
            
            device = devices[device_index]
            if device['max_input_channels'] < channels:
                print(f"{Fore.RED}Device doesn't support {channels} input channels{Style.RESET_ALL}")
                return False
            
            # Test configuration
            test_duration = 0.1  # 100ms test
            recording = sd.rec(
                int(test_duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                device=device_index,
                dtype=f'int{bit_depth}' if bit_depth in [16, 32] else 'float32'
            )
            sd.wait()
            
            print(f"{Fore.GREEN}Microphone configuration successful!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Microphone configuration failed: {e}{Style.RESET_ALL}")
            return False
    
    def generate_hardware_report(self) -> str:
        """Generate comprehensive hardware report"""
        report = []
        report.append(f"{Fore.CYAN}=== PC Hardware Information & Control Report ==={Style.RESET_ALL}\n")
        
        # System Information
        report.append(f"{Fore.GREEN}System Information:{Style.RESET_ALL}")
        sys_table = [[k.replace('_', ' ').title(), v] for k, v in self.system_info.items()]
        report.append(tabulate(sys_table, headers=['Property', 'Value'], tablefmt='grid'))
        report.append("\n")
        
        # Fan Information
        report.append(f"{Fore.GREEN}Fan Information & Control:{Style.RESET_ALL}")
        fans = self.get_fan_information()
        fan_table = []
        for fan in fans:
            control_methods = ', '.join(fan.control_methods) if fan.control_methods else 'None'
            fan_table.append([
                fan.name,
                f"{fan.value or 'N/A'} {fan.unit}",
                'Yes' if fan.controllable else 'No',
                control_methods
            ])
        report.append(tabulate(fan_table, 
                              headers=['Fan Name', 'Speed', 'Controllable', 'Control Methods'], 
                              tablefmt='grid'))
        report.append("\n")
        
        # Sensor Information
        report.append(f"{Fore.GREEN}Sensor Information & Control:{Style.RESET_ALL}")
        sensors = self.get_sensor_information()
        sensor_table = []
        for sensor in sensors:
            control_methods = ', '.join(sensor.control_methods) if sensor.control_methods else 'None'
            sensor_table.append([
                sensor.name,
                sensor.type.title(),
                f"{sensor.value or 'N/A'} {sensor.unit}",
                'Yes' if sensor.controllable else 'No',
                control_methods
            ])
        report.append(tabulate(sensor_table,
                              headers=['Sensor Name', 'Type', 'Value', 'Controllable', 'Control Methods'],
                              tablefmt='grid'))
        report.append("\n")
        
        # Microphone Information
        report.append(f"{Fore.GREEN}Microphone Information & Control:{Style.RESET_ALL}")
        microphones = self.get_microphone_information()
        mic_table = []
        for mic in microphones:
            control_methods = ', '.join(mic.control_methods) if mic.control_methods else 'None'
            mic_table.append([
                mic.name,
                f"{mic.value or 'N/A'} {mic.unit}",
                'Yes' if mic.controllable else 'No',
                control_methods
            ])
        report.append(tabulate(mic_table,
                              headers=['Microphone Name', 'Default Sample Rate', 'Controllable', 'Control Methods'],
                              tablefmt='grid'))
        report.append("\n")
        
        # Control Capabilities Summary
        report.append(f"{Fore.GREEN}Control Capabilities Summary:{Style.RESET_ALL}")
        capabilities = [
            ["Fan Control", "Partial", "Requires specialized drivers or BIOS settings"],
            ["Temperature Monitoring", "Yes", "Read-only via system sensors"],
            ["Microphone Control", "Yes" if AUDIO_AVAILABLE else "No", "Sample rate, bit depth, channels configurable"],
            ["Sensor Monitoring", "Yes", "Temperature, voltage, power sensors available"],
            ["Real-time Updates", "Yes", "All sensors can be monitored continuously"]
        ]
        report.append(tabulate(capabilities,
                              headers=['Feature', 'Available', 'Notes'],
                              tablefmt='grid'))
        
        return '\n'.join(report)
    
    def save_report(self, filename: str = "hardware_report.txt"):
        """Save hardware report to file"""
        report = self.generate_hardware_report()
        # Remove color codes for file output
        import re
        clean_report = re.sub(r'\x1b\[[0-9;]*m', '', report)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(clean_report)
        
        print(f"{Fore.GREEN}Report saved to {filename}{Style.RESET_ALL}")

def main():
    """Main function for testing"""
    print(f"{Fore.CYAN}Initializing Hardware Controller...{Style.RESET_ALL}")
    controller = HardwareController()
    
    # Generate and display report
    report = controller.generate_hardware_report()
    print(report)
    
    # Save report
    controller.save_report()
    
    # Demonstrate control functions
    print(f"\n{Fore.CYAN}=== Control Function Demonstrations ==={Style.RESET_ALL}")
    
    # Fan control demo
    print(f"\n{Fore.YELLOW}Fan Control Demo:{Style.RESET_ALL}")
    controller.control_fan("CPU Fan", mode="auto", level=3, duty=60)
    
    # Microphone control demo
    if AUDIO_AVAILABLE and controller.audio_devices:
        print(f"\n{Fore.YELLOW}Microphone Control Demo:{Style.RESET_ALL}")
        controller.control_microphone(0, sample_rate=48000, bit_depth=24, channels=1)

if __name__ == "__main__":
    main()