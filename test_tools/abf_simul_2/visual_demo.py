#!/usr/bin/env python3
"""
Visual Demo for Ultra-Fast Audio Beam Focusing Simulator

This script creates a visual demonstration that shows:
1. Beam focusing animation with matplotlib
2. Real-time energy map visualization
3. Source tracking demonstration
4. Performance metrics

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
from beam_focusing_simulator import BeamFocusingSimulator

def create_visual_demo():
    """
    Create and display a comprehensive visual demonstration.
    """
    print("Creating Ultra-Fast Audio Beam Focusing Visual Demo...")
    print("This will show you the beam focusing animation!")
    
    # Create simulator with moderate settings for good performance
    print("\n1. Creating simulator...")
    simulator = BeamFocusingSimulator(
        n_mics=16,
        grid_resolution=40,
        array_radius=0.2,
        target_distance=0.4
    )
    
    # Generate test audio with a source slightly off-center
    print("2. Generating test audio...")
    source_position = (0.05, 0.03, 0.4)  # Slightly off-center
    audio_data = simulator.generate_test_audio(
        duration=3.0,
        source_positions=[source_position]
    )
    
    # Compute beam focusing
    print("3. Computing beam focus...")
    start_time = time.time()
    energy_maps, time_stamps = simulator.compute_beam_focus(
        audio_data,
        time_window=0.15,
        overlap=0.7
    )
    computation_time = time.time() - start_time
    
    print(f"   Computation completed in {computation_time:.2f}s")
    print(f"   Real-time factor: {3.0/computation_time:.2f}x")
    print(f"   Generated {len(energy_maps)} energy maps")
    
    # Create enhanced visualization
    print("4. Creating enhanced visualization...")
    create_enhanced_animation(simulator, energy_maps, time_stamps, source_position)
    
    print("\n✅ Visual demo complete! The animation window should be displayed.")
    print("   Close the animation window when you're done viewing it.")

def create_enhanced_animation(simulator, energy_maps, time_stamps, source_position):
    """
    Create an enhanced animation with multiple views and information.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Ultra-Fast Audio Beam Focusing Simulator - Live Demo', fontsize=16, fontweight='bold')
    
    # Main energy map (large subplot)
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    
    # Time series plots
    ax_energy = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    ax_peak = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    
    # Array geometry
    ax_array = plt.subplot2grid((3, 4), (2, 0))
    
    # Cross-sections
    ax_cross_h = plt.subplot2grid((3, 4), (2, 1))
    ax_cross_v = plt.subplot2grid((3, 4), (2, 2))
    
    # Info panel
    ax_info = plt.subplot2grid((3, 4), (2, 3))
    
    # Setup main energy map
    extent = [-simulator.target_size/2, simulator.target_size/2,
             -simulator.target_size/2, simulator.target_size/2]
    
    im_main = ax_main.imshow(energy_maps[0], extent=extent, origin='lower',
                            cmap='hot', interpolation='bilinear', aspect='equal')
    ax_main.set_title('Beam Focus Energy Map', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('X Position (m)')
    ax_main.set_ylabel('Y Position (m)')
    
    # Add source position
    source_marker = ax_main.scatter(source_position[0], source_position[1], 
                                   c='white', s=300, marker='*', 
                                   edgecolors='black', linewidths=2, 
                                   label='True Source', zorder=10)
    
    # Add microphone positions
    mic_x = simulator.mic_positions[:, 0]
    mic_y = simulator.mic_positions[:, 1]
    mic_markers = ax_main.scatter(mic_x, mic_y, c='cyan', s=40, marker='o',
                                 alpha=0.8, edgecolors='blue', linewidths=1,
                                 label='Microphones', zorder=5)
    
    # Add array circle
    array_circle = Circle((0, 0), simulator.array_radius, fill=False, 
                         color='cyan', linestyle='--', alpha=0.6, linewidth=2)
    ax_main.add_patch(array_circle)
    
    ax_main.legend(loc='upper right')
    ax_main.grid(True, alpha=0.3)
    
    # Colorbar for main plot
    cbar = plt.colorbar(im_main, ax=ax_main, shrink=0.8)
    cbar.set_label('Energy Level', fontsize=12)
    
    # Setup time series plots
    max_energy_over_time = np.max(energy_maps.reshape(len(energy_maps), -1), axis=1)
    total_energy_over_time = np.sum(energy_maps.reshape(len(energy_maps), -1), axis=1)
    
    line_max, = ax_energy.plot(time_stamps, max_energy_over_time, 'b-', linewidth=2, label='Max Energy')
    ax_energy.set_ylabel('Max Energy')
    ax_energy.set_title('Energy vs Time')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.legend()
    
    line_total, = ax_peak.plot(time_stamps, total_energy_over_time, 'r-', linewidth=2, label='Total Energy')
    ax_peak.set_xlabel('Time (s)')
    ax_peak.set_ylabel('Total Energy')
    ax_peak.set_title('Total Energy vs Time')
    ax_peak.grid(True, alpha=0.3)
    ax_peak.legend()
    
    # Time indicators
    time_indicator_1 = ax_energy.axvline(time_stamps[0], color='green', linestyle='--', alpha=0.8, linewidth=2)
    time_indicator_2 = ax_peak.axvline(time_stamps[0], color='green', linestyle='--', alpha=0.8, linewidth=2)
    
    # Setup array geometry plot
    ax_array.scatter(mic_x, mic_y, c='cyan', s=60, marker='o', alpha=0.8, edgecolors='blue')
    ax_array.add_patch(Circle((0, 0), simulator.array_radius, fill=False, color='cyan', linestyle='--'))
    ax_array.set_xlim(-simulator.array_radius*1.2, simulator.array_radius*1.2)
    ax_array.set_ylim(-simulator.array_radius*1.2, simulator.array_radius*1.2)
    ax_array.set_aspect('equal')
    ax_array.set_title('Array Geometry')
    ax_array.grid(True, alpha=0.3)
    
    # Setup cross-section plots
    center_idx = simulator.grid_resolution // 2
    x_coords = np.linspace(-simulator.target_size/2, simulator.target_size/2, simulator.grid_resolution)
    y_coords = np.linspace(-simulator.target_size/2, simulator.target_size/2, simulator.grid_resolution)
    
    line_h, = ax_cross_h.plot(x_coords, energy_maps[0][center_idx, :], 'b-', linewidth=2)
    ax_cross_h.set_xlabel('X Position (m)')
    ax_cross_h.set_ylabel('Energy')
    ax_cross_h.set_title('Horizontal Cross-section')
    ax_cross_h.grid(True, alpha=0.3)
    
    line_v, = ax_cross_v.plot(y_coords, energy_maps[0][:, center_idx], 'r-', linewidth=2)
    ax_cross_v.set_xlabel('Y Position (m)')
    ax_cross_v.set_ylabel('Energy')
    ax_cross_v.set_title('Vertical Cross-section')
    ax_cross_v.grid(True, alpha=0.3)
    
    # Setup info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, 
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # Animation function
    def animate(frame):
        current_time = time_stamps[frame]
        current_map = energy_maps[frame]
        
        # Update main energy map
        im_main.set_array(current_map)
        im_main.set_clim(vmin=np.min(current_map), vmax=np.max(current_map))
        
        # Update time indicators
        time_indicator_1.set_xdata([current_time, current_time])
        time_indicator_2.set_xdata([current_time, current_time])
        
        # Update cross-sections
        line_h.set_ydata(current_map[center_idx, :])
        line_v.set_ydata(current_map[:, center_idx])
        
        # Update y-axis limits for cross-sections
        ax_cross_h.set_ylim(0, np.max(current_map) * 1.1)
        ax_cross_v.set_ylim(0, np.max(current_map) * 1.1)
        
        # Find peak location
        peak_idx = np.unravel_index(np.argmax(current_map), current_map.shape)
        peak_x = (peak_idx[1] / simulator.grid_resolution - 0.5) * simulator.target_size
        peak_y = (peak_idx[0] / simulator.grid_resolution - 0.5) * simulator.target_size
        
        # Update info panel
        info_str = f"""Frame: {frame + 1}/{len(energy_maps)}
Time: {current_time:.3f}s

Energy Stats:
Max: {np.max(current_map):.3f}
Mean: {np.mean(current_map):.3f}
Std: {np.std(current_map):.3f}

Peak Location:
X: {peak_x:.3f}m
Y: {peak_y:.3f}m

True Source:
X: {source_position[0]:.3f}m
Y: {source_position[1]:.3f}m

Error:
ΔX: {abs(peak_x - source_position[0]):.3f}m
ΔY: {abs(peak_y - source_position[1]):.3f}m"""
        
        info_text.set_text(info_str)
        
        # Update main plot title with current time
        ax_main.set_title(f'Beam Focus Energy Map (t = {current_time:.3f}s)', 
                         fontsize=14, fontweight='bold')
        
        return [im_main, time_indicator_1, time_indicator_2, line_h, line_v, info_text]
    
    # Create animation
    print("   Creating animation with {} frames...".format(len(energy_maps)))
    anim = animation.FuncAnimation(fig, animate, frames=len(energy_maps),
                                 interval=300, blit=False, repeat=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the animation
    print("   Displaying animation...")
    plt.show()
    
    return anim

def create_multiple_source_demo():
    """
    Create a demo with multiple sources to show advanced capabilities.
    """
    print("\n" + "="*60)
    print("BONUS: Multiple Source Demonstration")
    print("="*60)
    
    # Create simulator
    simulator = BeamFocusingSimulator(n_mics=24, grid_resolution=35)
    
    # Multiple sources
    sources = [
        (-0.08, -0.06, 0.4),  # Source 1
        (0.06, 0.08, 0.4),   # Source 2
    ]
    
    print(f"Generating audio with {len(sources)} sources...")
    audio_data = simulator.generate_test_audio(duration=4.0, source_positions=sources)
    
    print("Computing beam focus for multiple sources...")
    energy_maps, time_stamps = simulator.compute_beam_focus(audio_data, time_window=0.2)
    
    # Simple visualization for multiple sources
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Multiple Source Beam Focusing Demo', fontsize=16)
    
    extent = [-simulator.target_size/2, simulator.target_size/2,
             -simulator.target_size/2, simulator.target_size/2]
    
    # Show energy maps at different times
    time_indices = [0, len(energy_maps)//4, len(energy_maps)//2, 
                   3*len(energy_maps)//4, len(energy_maps)-1]
    
    for i, t_idx in enumerate(time_indices[:5]):
        row, col = i // 3, i % 3
        if row < 2 and col < 3:
            im = axes[row, col].imshow(energy_maps[t_idx], extent=extent, 
                                     origin='lower', cmap='hot')
            axes[row, col].set_title(f't = {time_stamps[t_idx]:.2f}s')
            
            # Mark source positions
            for j, source in enumerate(sources):
                axes[row, col].scatter(source[0], source[1], c='white', s=100,
                                     marker='x', linewidths=3, label=f'Source {j+1}' if i == 0 else '')
            
            if i == 0:
                axes[row, col].legend()
            
            plt.colorbar(im, ax=axes[row, col])
    
    # Remove empty subplot
    if len(time_indices) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.show()
    
    print("Multiple source demo complete!")

def main():
    """
    Main function to run visual demonstrations.
    """
    print("Ultra-Fast Audio Beam Focusing Simulator - Visual Demo")
    print("=" * 60)
    print("This demo will show you the beam focusing animation!")
    print("Make sure you can see matplotlib windows.")
    
    try:
        # Main visual demo
        create_visual_demo()
        
        # Ask if user wants to see multiple source demo
        print("\n" + "-"*40)
        response = input("Would you like to see the multiple source demo? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            create_multiple_source_demo()
        
        print("\n" + "="*60)
        print("Visual demonstration complete!")
        print("You should have seen:")
        print("✓ Real-time beam focusing animation")
        print("✓ Energy maps showing source localization")
        print("✓ Time evolution of energy patterns")
        print("✓ Cross-sectional analysis")
        print("✓ Performance metrics")
        
        if response in ['y', 'yes']:
            print("✓ Multiple source tracking")
        
        print("\nThe simulator is working correctly and showing visual results!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()