#!/usr/bin/env python3
"""
Hardware Controller GUI Application
Provides a user-friendly interface for hardware monitoring and control

Author: Hardware Control Project
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from hardware_controller import HardwareController, AUDIO_AVAILABLE
from typing import Dict, List

class HardwareControllerGUI:
    """GUI Application for Hardware Controller"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PC Hardware Controller")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize hardware controller
        self.controller = HardwareController()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.setup_ui()
        self.refresh_data()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_fan_control_tab()
        self.create_sensor_tab()
        self.create_microphone_tab()
        self.create_report_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_overview_tab(self):
        """Create system overview tab"""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="System Overview")
        
        # System info frame
        sys_frame = ttk.LabelFrame(overview_frame, text="System Information")
        sys_frame.pack(fill='x', padx=10, pady=5)
        
        self.sys_tree = ttk.Treeview(sys_frame, columns=('Value',), show='tree headings')
        self.sys_tree.heading('#0', text='Property')
        self.sys_tree.heading('Value', text='Value')
        self.sys_tree.pack(fill='x', padx=5, pady=5)
        
        # Quick stats frame
        stats_frame = ttk.LabelFrame(overview_frame, text="Quick Statistics")
        stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create stats labels
        self.stats_labels = {}
        stats_info = [
            ('CPU Usage', 'cpu_percent'),
            ('Memory Usage', 'memory_percent'),
            ('Disk Usage', 'disk_percent'),
            ('Network Sent', 'network_sent'),
            ('Network Received', 'network_recv')
        ]
        
        for i, (label, key) in enumerate(stats_info):
            row = i // 2
            col = i % 2
            
            frame = ttk.Frame(stats_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
            
            ttk.Label(frame, text=f"{label}:", font=('Arial', 10, 'bold')).pack(anchor='w')
            self.stats_labels[key] = ttk.Label(frame, text="Loading...", font=('Arial', 12))
            self.stats_labels[key].pack(anchor='w')
        
        # Configure grid weights
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        
        # Control buttons
        button_frame = ttk.Frame(overview_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh Data", command=self.refresh_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Start Monitoring", command=self.start_monitoring).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop Monitoring", command=self.stop_monitoring).pack(side='left', padx=5)
    
    def create_fan_control_tab(self):
        """Create fan control tab"""
        fan_frame = ttk.Frame(self.notebook)
        self.notebook.add(fan_frame, text="Fan Control")
        
        # Fan information
        info_frame = ttk.LabelFrame(fan_frame, text="Fan Information")
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.fan_tree = ttk.Treeview(info_frame, columns=('Speed', 'Controllable', 'Methods'), show='tree headings')
        self.fan_tree.heading('#0', text='Fan Name')
        self.fan_tree.heading('Speed', text='Speed (RPM)')
        self.fan_tree.heading('Controllable', text='Controllable')
        self.fan_tree.heading('Methods', text='Control Methods')
        self.fan_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Fan control panel
        control_frame = ttk.LabelFrame(fan_frame, text="Fan Control Panel")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Control options
        ttk.Label(control_frame, text="Fan Mode:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.fan_mode_var = tk.StringVar(value="auto")
        fan_mode_combo = ttk.Combobox(control_frame, textvariable=self.fan_mode_var, 
                                     values=["auto", "manual", "silent", "performance"], state="readonly")
        fan_mode_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Label(control_frame, text="Fan Level (1-10):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.fan_level_var = tk.IntVar(value=5)
        fan_level_scale = ttk.Scale(control_frame, from_=1, to=10, variable=self.fan_level_var, orient='horizontal')
        fan_level_scale.grid(row=0, column=3, padx=5, pady=5, sticky='ew')
        
        ttk.Label(control_frame, text="PWM Duty (%):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.fan_duty_var = tk.IntVar(value=50)
        fan_duty_scale = ttk.Scale(control_frame, from_=0, to=100, variable=self.fan_duty_var, orient='horizontal')
        fan_duty_scale.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Button(control_frame, text="Apply Fan Settings", command=self.apply_fan_settings).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)
        
        # Information text
        info_text = tk.Text(fan_frame, height=6, wrap=tk.WORD)
        info_text.pack(fill='x', padx=10, pady=5)
        info_text.insert('1.0', 
            "Fan Control Information:\n"
            "• Most modern systems require BIOS/UEFI configuration for fan control\n"
            "• Some motherboards provide software utilities for fan management\n"
            "• PWM (Pulse Width Modulation) is the standard method for fan speed control\n"
            "• Administrative privileges may be required for direct hardware access\n"
            "• Consider using manufacturer-specific software for best compatibility"
        )
        info_text.config(state='disabled')
    
    def create_sensor_tab(self):
        """Create sensor monitoring tab"""
        sensor_frame = ttk.Frame(self.notebook)
        self.notebook.add(sensor_frame, text="Sensors")
        
        # Sensor information
        self.sensor_tree = ttk.Treeview(sensor_frame, columns=('Type', 'Value', 'Unit', 'Range'), show='tree headings')
        self.sensor_tree.heading('#0', text='Sensor Name')
        self.sensor_tree.heading('Type', text='Type')
        self.sensor_tree.heading('Value', text='Current Value')
        self.sensor_tree.heading('Unit', text='Unit')
        self.sensor_tree.heading('Range', text='Range')
        self.sensor_tree.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Sensor control options
        sensor_control_frame = ttk.LabelFrame(sensor_frame, text="Sensor Settings")
        sensor_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(sensor_control_frame, text="Temperature Alert Threshold (°C):").pack(side='left', padx=5)
        self.temp_threshold_var = tk.IntVar(value=80)
        temp_threshold_scale = ttk.Scale(sensor_control_frame, from_=50, to=100, 
                                       variable=self.temp_threshold_var, orient='horizontal')
        temp_threshold_scale.pack(side='left', padx=5, fill='x', expand=True)
        
        ttk.Button(sensor_control_frame, text="Set Alert", command=self.set_temperature_alert).pack(side='right', padx=5)
    
    def create_microphone_tab(self):
        """Create microphone control tab"""
        mic_frame = ttk.Frame(self.notebook)
        self.notebook.add(mic_frame, text="Microphone")
        
        if not AUDIO_AVAILABLE:
            ttk.Label(mic_frame, text="Audio libraries not available. Please install pyaudio and sounddevice.", 
                     font=('Arial', 12)).pack(expand=True)
            return
        
        # Microphone information
        info_frame = ttk.LabelFrame(mic_frame, text="Available Microphones")
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.mic_tree = ttk.Treeview(info_frame, columns=('Default_Rate', 'Channels'), show='tree headings')
        self.mic_tree.heading('#0', text='Microphone Name')
        self.mic_tree.heading('Default_Rate', text='Default Sample Rate')
        self.mic_tree.heading('Channels', text='Max Channels')
        self.mic_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Microphone control panel
        control_frame = ttk.LabelFrame(mic_frame, text="Microphone Settings")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Device selection
        ttk.Label(control_frame, text="Device:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.mic_device_var = tk.IntVar(value=0)
        self.mic_device_combo = ttk.Combobox(control_frame, state="readonly")
        self.mic_device_combo.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Sample rate
        ttk.Label(control_frame, text="Sample Rate (Hz):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.sample_rate_var = tk.StringVar(value="44100")
        sample_rate_combo = ttk.Combobox(control_frame, textvariable=self.sample_rate_var,
                                       values=["8000", "16000", "22050", "44100", "48000", "96000", "192000"],
                                       state="readonly")
        sample_rate_combo.grid(row=0, column=3, padx=5, pady=5, sticky='ew')
        
        # Bit depth
        ttk.Label(control_frame, text="Bit Depth:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.bit_depth_var = tk.StringVar(value="16")
        bit_depth_combo = ttk.Combobox(control_frame, textvariable=self.bit_depth_var,
                                     values=["16", "24", "32"], state="readonly")
        bit_depth_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        
        # Channels
        ttk.Label(control_frame, text="Channels:").grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.channels_var = tk.StringVar(value="1")
        channels_combo = ttk.Combobox(control_frame, textvariable=self.channels_var,
                                    values=["1", "2"], state="readonly")
        channels_combo.grid(row=1, column=3, padx=5, pady=5, sticky='ew')
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Test Configuration", command=self.test_microphone).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Record 3 Seconds", command=self.record_test).pack(side='left', padx=5)
        
        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)
    
    def create_report_tab(self):
        """Create report generation tab"""
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="Report")
        
        # Report display
        self.report_text = scrolledtext.ScrolledText(report_frame, wrap=tk.WORD, font=('Courier', 10))
        self.report_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Report controls
        button_frame = ttk.Frame(report_frame)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Report", command=self.save_report).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_report).pack(side='left', padx=5)
    
    def refresh_data(self):
        """Refresh all hardware data"""
        self.update_status("Refreshing data...")
        
        # Update system info
        self.update_system_info()
        
        # Update quick stats
        self.update_quick_stats()
        
        # Update fan info
        self.update_fan_info()
        
        # Update sensor info
        self.update_sensor_info()
        
        # Update microphone info
        if AUDIO_AVAILABLE:
            self.update_microphone_info()
        
        self.update_status("Data refreshed")
    
    def update_system_info(self):
        """Update system information display"""
        # Clear existing items
        for item in self.sys_tree.get_children():
            self.sys_tree.delete(item)
        
        # Add system info
        for key, value in self.controller.system_info.items():
            display_key = key.replace('_', ' ').title()
            self.sys_tree.insert('', 'end', text=display_key, values=(str(value),))
    
    def update_quick_stats(self):
        """Update quick statistics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.stats_labels['cpu_percent'].config(text=f"{cpu_percent:.1f}%")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.stats_labels['memory_percent'].config(text=f"{memory.percent:.1f}%")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.stats_labels['disk_percent'].config(text=f"{disk_percent:.1f}%")
            
            # Network stats
            net_io = psutil.net_io_counters()
            self.stats_labels['network_sent'].config(text=f"{net_io.bytes_sent / (1024*1024):.1f} MB")
            self.stats_labels['network_recv'].config(text=f"{net_io.bytes_recv / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def update_fan_info(self):
        """Update fan information display"""
        # Clear existing items
        for item in self.fan_tree.get_children():
            self.fan_tree.delete(item)
        
        # Add fan info
        fans = self.controller.get_fan_information()
        for fan in fans:
            speed = f"{fan.value or 'N/A'}"
            controllable = "Yes" if fan.controllable else "No"
            methods = ", ".join(fan.control_methods) if fan.control_methods else "None"
            self.fan_tree.insert('', 'end', text=fan.name, values=(speed, controllable, methods))
    
    def update_sensor_info(self):
        """Update sensor information display"""
        # Clear existing items
        for item in self.sensor_tree.get_children():
            self.sensor_tree.delete(item)
        
        # Add sensor info
        sensors = self.controller.get_sensor_information()
        for sensor in sensors:
            value = f"{sensor.value or 'N/A'}"
            range_str = ""
            if sensor.min_value is not None and sensor.max_value is not None:
                range_str = f"{sensor.min_value}-{sensor.max_value}"
            
            self.sensor_tree.insert('', 'end', text=sensor.name, 
                                  values=(sensor.type.title(), value, sensor.unit, range_str))
    
    def update_microphone_info(self):
        """Update microphone information display"""
        if not AUDIO_AVAILABLE:
            return
        
        # Clear existing items
        for item in self.mic_tree.get_children():
            self.mic_tree.delete(item)
        
        # Update device combo
        device_names = []
        
        # Add microphone info
        microphones = self.controller.get_microphone_information()
        for i, mic in enumerate(microphones):
            if mic.type == "microphone":
                device_names.append(f"{i}: {mic.name}")
                rate = f"{mic.value or 'N/A'}"
                # Get channel info from audio devices
                channels = "N/A"
                if i < len(self.controller.audio_devices):
                    channels = str(self.controller.audio_devices[i]['channels'])
                
                self.mic_tree.insert('', 'end', text=mic.name, values=(rate, channels))
        
        # Update device combo
        if hasattr(self, 'mic_device_combo'):
            self.mic_device_combo['values'] = device_names
            if device_names:
                self.mic_device_combo.current(0)
    
    def apply_fan_settings(self):
        """Apply fan control settings"""
        selected_item = self.fan_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a fan first")
            return
        
        fan_name = self.fan_tree.item(selected_item[0])['text']
        mode = self.fan_mode_var.get()
        level = self.fan_level_var.get()
        duty = self.fan_duty_var.get()
        
        success = self.controller.control_fan(fan_name, mode=mode, level=level, duty=duty)
        
        if success:
            messagebox.showinfo("Success", "Fan settings applied successfully")
        else:
            messagebox.showinfo("Information", 
                              "Fan control demonstration completed. "
                              "Check console for detailed information about control methods.")
    
    def test_microphone(self):
        """Test microphone configuration"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not available")
            return
        
        try:
            device_index = int(self.mic_device_combo.current())
            sample_rate = int(self.sample_rate_var.get())
            bit_depth = int(self.bit_depth_var.get())
            channels = int(self.channels_var.get())
            
            success = self.controller.control_microphone(device_index, sample_rate, bit_depth, channels)
            
            if success:
                messagebox.showinfo("Success", "Microphone configuration test successful")
            else:
                messagebox.showerror("Error", "Microphone configuration test failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Test failed: {str(e)}")
    
    def record_test(self):
        """Record a 3-second test audio"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio libraries not available")
            return
        
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write
            
            device_index = int(self.mic_device_combo.current())
            sample_rate = int(self.sample_rate_var.get())
            channels = int(self.channels_var.get())
            
            messagebox.showinfo("Recording", "Recording will start in 1 second. Speak into the microphone.")
            
            # Record for 3 seconds
            duration = 3
            recording = sd.rec(int(duration * sample_rate), 
                             samplerate=sample_rate, 
                             channels=channels,
                             device=device_index)
            sd.wait()
            
            # Save recording
            filename = "test_recording.wav"
            write(filename, sample_rate, recording)
            
            messagebox.showinfo("Success", f"Recording saved as {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Recording failed: {str(e)}")
    
    def set_temperature_alert(self):
        """Set temperature alert threshold"""
        threshold = self.temp_threshold_var.get()
        messagebox.showinfo("Alert Set", f"Temperature alert threshold set to {threshold}°C")
        # In a real implementation, this would set up monitoring alerts
    
    def generate_report(self):
        """Generate hardware report"""
        self.update_status("Generating report...")
        report = self.controller.generate_hardware_report()
        
        # Remove color codes for GUI display
        import re
        clean_report = re.sub(r'\x1b\[[0-9;]*m', '', report)
        
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', clean_report)
        self.update_status("Report generated")
    
    def save_report(self):
        """Save report to file"""
        try:
            self.controller.save_report()
            messagebox.showinfo("Success", "Report saved to hardware_report.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")
    
    def clear_report(self):
        """Clear report display"""
        self.report_text.delete('1.0', tk.END)
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.update_status("Monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        self.update_status("Monitoring stopped")
    
    def monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                self.root.after(0, self.update_quick_stats)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()

def main():
    """Main function"""
    root = tk.Tk()
    app = HardwareControllerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()