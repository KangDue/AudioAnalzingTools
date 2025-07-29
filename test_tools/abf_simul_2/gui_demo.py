#!/usr/bin/env python3
"""
GUI Demo for Ultra-Fast Audio Beam Focusing Simulator

This script creates a visual GUI demonstration that shows:
1. Real-time beam focusing animation
2. Interactive controls for parameters
3. Live energy map visualization
4. Performance metrics display

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from beam_focusing_simulator import BeamFocusingSimulator

class BeamFocusingGUI:
    """
    GUI application for interactive beam focusing demonstration.
    """
    
    def __init__(self):
        self.simulator = None
        self.energy_maps = None
        self.time_stamps = None
        self.animation = None
        self.is_running = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Ultra-Fast Audio Beam Focusing Simulator")
        self.root.geometry("800x600")
        
        # Setup GUI components
        self.setup_gui()
        
        # Initialize with default parameters
        self.create_simulator()
    
    def setup_gui(self):
        """
        Setup the GUI layout and components.
        """
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Ultra-Fast Audio Beam Focusing Simulator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Parameter controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Simulator Parameters", padding="10")
        controls_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Number of microphones
        ttk.Label(controls_frame, text="Microphones:").grid(row=0, column=0, sticky=tk.W)
        self.n_mics_var = tk.IntVar(value=16)
        n_mics_combo = ttk.Combobox(controls_frame, textvariable=self.n_mics_var, 
                                   values=[8, 16, 24, 32], state="readonly", width=10)
        n_mics_combo.grid(row=0, column=1, padx=(5, 20))
        
        # Grid resolution
        ttk.Label(controls_frame, text="Grid Resolution:").grid(row=0, column=2, sticky=tk.W)
        self.grid_res_var = tk.IntVar(value=30)
        grid_res_combo = ttk.Combobox(controls_frame, textvariable=self.grid_res_var,
                                     values=[20, 30, 40, 50], state="readonly", width=10)
        grid_res_combo.grid(row=0, column=3, padx=(5, 20))
        
        # Array radius
        ttk.Label(controls_frame, text="Array Radius (m):").grid(row=1, column=0, sticky=tk.W)
        self.array_radius_var = tk.DoubleVar(value=0.2)
        array_radius_spin = ttk.Spinbox(controls_frame, from_=0.1, to=0.5, increment=0.05,
                                       textvariable=self.array_radius_var, width=10)
        array_radius_spin.grid(row=1, column=1, padx=(5, 20))
        
        # Target distance
        ttk.Label(controls_frame, text="Target Distance (m):").grid(row=1, column=2, sticky=tk.W)
        self.target_dist_var = tk.DoubleVar(value=0.4)
        target_dist_spin = ttk.Spinbox(controls_frame, from_=0.2, to=1.0, increment=0.1,
                                      textvariable=self.target_dist_var, width=10)
        target_dist_spin.grid(row=1, column=3, padx=(5, 20))
        
        # Source position controls
        source_frame = ttk.LabelFrame(main_frame, text="Source Position", padding="10")
        source_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(source_frame, text="X Position (m):").grid(row=0, column=0, sticky=tk.W)
        self.source_x_var = tk.DoubleVar(value=0.0)
        source_x_spin = ttk.Spinbox(source_frame, from_=-0.15, to=0.15, increment=0.02,
                                   textvariable=self.source_x_var, width=10)
        source_x_spin.grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(source_frame, text="Y Position (m):").grid(row=0, column=2, sticky=tk.W)
        self.source_y_var = tk.DoubleVar(value=0.0)
        source_y_spin = ttk.Spinbox(source_frame, from_=-0.15, to=0.15, increment=0.02,
                                   textvariable=self.source_y_var, width=10)
        source_y_spin.grid(row=0, column=3, padx=(5, 20))
        
        # Audio parameters
        audio_frame = ttk.LabelFrame(main_frame, text="Audio Parameters", padding="10")
        audio_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(audio_frame, text="Duration (s):").grid(row=0, column=0, sticky=tk.W)
        self.duration_var = tk.DoubleVar(value=2.0)
        duration_spin = ttk.Spinbox(audio_frame, from_=1.0, to=10.0, increment=0.5,
                                   textvariable=self.duration_var, width=10)
        duration_spin.grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(audio_frame, text="Time Window (s):").grid(row=0, column=2, sticky=tk.W)
        self.time_window_var = tk.DoubleVar(value=0.2)
        time_window_spin = ttk.Spinbox(audio_frame, from_=0.1, to=0.5, increment=0.05,
                                      textvariable=self.time_window_var, width=10)
        time_window_spin.grid(row=0, column=3, padx=(5, 20))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.update_btn = ttk.Button(button_frame, text="Update Simulator", 
                                    command=self.create_simulator)
        self.update_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Audio", 
                                      command=self.generate_audio)
        self.generate_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.compute_btn = ttk.Button(button_frame, text="Compute Beam Focus", 
                                     command=self.compute_beam_focus)
        self.compute_btn.grid(row=0, column=2, padx=(0, 10))
        
        self.visualize_btn = ttk.Button(button_frame, text="Show Animation", 
                                       command=self.show_animation)
        self.visualize_btn.grid(row=0, column=3, padx=(0, 10))
        
        # Status and progress
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready - Click 'Update Simulator' to begin")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
    
    def create_simulator(self):
        """
        Create simulator with current parameters.
        """
        try:
            self.status_var.set("Creating simulator...")
            self.progress.start()
            
            self.simulator = BeamFocusingSimulator(
                n_mics=self.n_mics_var.get(),
                array_radius=self.array_radius_var.get(),
                target_distance=self.target_dist_var.get(),
                grid_resolution=self.grid_res_var.get()
            )
            
            self.progress.stop()
            self.status_var.set(f"Simulator created: {self.n_mics_var.get()} mics, "
                               f"{self.grid_res_var.get()}×{self.grid_res_var.get()} grid")
            
            # Enable next step
            self.generate_btn.config(state='normal')
            
        except Exception as e:
            self.progress.stop()
            self.status_var.set(f"Error creating simulator: {str(e)}")
            messagebox.showerror("Error", f"Failed to create simulator: {str(e)}")
    
    def generate_audio(self):
        """
        Generate audio with current source position.
        """
        if not self.simulator:
            messagebox.showwarning("Warning", "Please create simulator first!")
            return
        
        try:
            self.status_var.set("Generating audio...")
            self.progress.start()
            
            source_pos = (self.source_x_var.get(), self.source_y_var.get(), 
                         self.target_dist_var.get())
            
            self.audio_data = self.simulator.generate_test_audio(
                duration=self.duration_var.get(),
                source_positions=[source_pos]
            )
            
            self.progress.stop()
            self.status_var.set(f"Audio generated: {self.audio_data.shape[0]} samples, "
                               f"{self.audio_data.shape[1]} channels")
            
            # Enable next step
            self.compute_btn.config(state='normal')
            
        except Exception as e:
            self.progress.stop()
            self.status_var.set(f"Error generating audio: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate audio: {str(e)}")
    
    def compute_beam_focus(self):
        """
        Compute beam focusing in a separate thread.
        """
        if not hasattr(self, 'audio_data'):
            messagebox.showwarning("Warning", "Please generate audio first!")
            return
        
        # Run computation in separate thread to avoid GUI freezing
        thread = threading.Thread(target=self._compute_beam_focus_thread)
        thread.daemon = True
        thread.start()
    
    def _compute_beam_focus_thread(self):
        """
        Thread function for beam focus computation.
        """
        try:
            self.root.after(0, lambda: self.status_var.set("Computing beam focus..."))
            self.root.after(0, lambda: self.progress.start())
            
            start_time = time.time()
            
            self.energy_maps, self.time_stamps = self.simulator.compute_beam_focus(
                self.audio_data,
                time_window=self.time_window_var.get(),
                overlap=0.5
            )
            
            computation_time = time.time() - start_time
            rt_factor = self.duration_var.get() / computation_time
            
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_var.set(
                f"Beam focus computed: {len(self.energy_maps)} maps, "
                f"RT factor: {rt_factor:.2f}x"))
            
            # Enable visualization
            self.root.after(0, lambda: self.visualize_btn.config(state='normal'))
            
        except Exception as e:
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.status_var.set(f"Error computing: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Computation failed: {str(e)}"))
    
    def show_animation(self):
        """
        Show the beam focusing animation in a new window.
        """
        if self.energy_maps is None:
            messagebox.showwarning("Warning", "Please compute beam focus first!")
            return
        
        try:
            self.status_var.set("Creating animation...")
            
            # Create animation window
            self.create_animation_window()
            
            self.status_var.set("Animation displayed! Close animation window when done.")
            
        except Exception as e:
            self.status_var.set(f"Error creating animation: {str(e)}")
            messagebox.showerror("Error", f"Failed to create animation: {str(e)}")
    
    def create_animation_window(self):
        """
        Create a separate window with the beam focusing animation.
        """
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Ultra-Fast Audio Beam Focusing - Real-Time Animation', fontsize=14)
        
        # Setup energy map plot
        extent = [-self.simulator.target_size/2, self.simulator.target_size/2,
                 -self.simulator.target_size/2, self.simulator.target_size/2]
        
        # Initial energy map
        im = ax1.imshow(self.energy_maps[0], extent=extent, origin='lower',
                       cmap='hot', interpolation='bilinear')
        ax1.set_title('Beam Focus Energy Map')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        
        # Add source position marker
        source_x = self.source_x_var.get()
        source_y = self.source_y_var.get()
        ax1.scatter(source_x, source_y, c='white', s=200, marker='x', 
                   linewidths=4, label='Source')
        
        # Add microphone positions
        mic_x = self.simulator.mic_positions[:, 0]
        mic_y = self.simulator.mic_positions[:, 1]
        ax1.scatter(mic_x, mic_y, c='cyan', s=30, marker='o', 
                   alpha=0.7, label='Microphones')
        
        ax1.legend()
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Energy')
        
        # Setup time series plot
        max_energy_over_time = np.max(self.energy_maps.reshape(len(self.energy_maps), -1), axis=1)
        line, = ax2.plot(self.time_stamps, max_energy_over_time, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Max Energy')
        ax2.set_title('Maximum Energy Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Add current time indicator
        time_line = ax2.axvline(self.time_stamps[0], color='red', linestyle='--', 
                               alpha=0.8, linewidth=2)
        
        # Animation function
        def animate(frame):
            # Update energy map
            im.set_array(self.energy_maps[frame])
            im.set_clim(vmin=np.min(self.energy_maps[frame]), 
                       vmax=np.max(self.energy_maps[frame]))
            
            # Update time indicator
            time_line.set_xdata([self.time_stamps[frame], self.time_stamps[frame]])
            
            # Update title with current time
            ax1.set_title(f'Beam Focus Energy Map (t = {self.time_stamps[frame]:.3f}s)')
            
            return [im, time_line]
        
        # Create and start animation
        anim = animation.FuncAnimation(fig, animate, frames=len(self.energy_maps),
                                     interval=200, blit=False, repeat=True)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def run(self):
        """
        Start the GUI application.
        """
        try:
            print("Starting Ultra-Fast Audio Beam Focusing Simulator GUI...")
            print("\nInstructions:")
            print("1. Adjust parameters as needed")
            print("2. Click 'Update Simulator' to create simulator")
            print("3. Click 'Generate Audio' to create test audio")
            print("4. Click 'Compute Beam Focus' to process")
            print("5. Click 'Show Animation' to see results!")
            print("\nClose the main window to exit.")
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("\nGUI interrupted by user.")
        except Exception as e:
            print(f"\nGUI error: {e}")
            messagebox.showerror("Fatal Error", f"GUI crashed: {str(e)}")


def main():
    """
    Main function to run the GUI demo.
    """
    print("="*60)
    print("Ultra-Fast Audio Beam Focusing Simulator - GUI Demo")
    print("="*60)
    
    try:
        # Create and run GUI
        app = BeamFocusingGUI()
        app.run()
        
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        print("\nTrying fallback matplotlib demo...")
        
        # Fallback to simple matplotlib demo
        fallback_demo()


def fallback_demo():
    """
    Fallback demo using matplotlib directly if GUI fails.
    """
    print("\nRunning fallback matplotlib demo...")
    
    try:
        # Create simulator
        simulator = BeamFocusingSimulator(n_mics=16, grid_resolution=30)
        print("Simulator created successfully.")
        
        # Generate audio
        print("Generating test audio...")
        audio_data = simulator.generate_test_audio(duration=2.0)
        
        # Compute beam focusing
        print("Computing beam focus...")
        energy_maps, time_stamps = simulator.compute_beam_focus(audio_data)
        
        # Show animation
        print("Creating animation...")
        anim = simulator.visualize_energy_maps(energy_maps, time_stamps)
        
        print("\n✅ Animation should now be displayed!")
        print("Close the animation window when done.")
        
    except Exception as e:
        print(f"Fallback demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()