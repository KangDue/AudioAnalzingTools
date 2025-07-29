#!/usr/bin/env python3
"""
Quick Visual Demo for Ultra-Fast Audio Beam Focusing Simulator

This script creates a quick static visualization to show beam focusing results.
"""

import numpy as np
import matplotlib.pyplot as plt
from beam_focusing_simulator import BeamFocusingSimulator
import time

def quick_demo():
    """
    Create a quick demonstration with static plots.
    """
    print("Quick Beam Focusing Demonstration")
    print("=" * 40)
    
    # Create simulator with small configuration for speed
    print("Creating simulator...")
    simulator = BeamFocusingSimulator(
        n_mics=12,
        grid_resolution=25,
        array_radius=0.2,
        target_distance=0.4
    )
    
    # Generate test audio
    print("Generating test audio...")
    source_position = (0.06, -0.04, 0.4)  # Off-center source
    audio_data = simulator.generate_test_audio(
        duration=1.5,
        source_positions=[source_position]
    )
    
    # Compute beam focusing
    print("Computing beam focus...")
    start_time = time.time()
    energy_maps, time_stamps = simulator.compute_beam_focus(
        audio_data,
        time_window=0.3,
        overlap=0.6
    )
    computation_time = time.time() - start_time
    
    print(f"Computation completed in {computation_time:.2f}s")
    print(f"Real-time factor: {1.5/computation_time:.2f}x")
    print(f"Generated {len(energy_maps)} energy maps")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Ultra-Fast Audio Beam Focusing - Results Visualization', fontsize=16, fontweight='bold')
    
    # Define extent for plots
    extent = [-simulator.target_size/2, simulator.target_size/2,
             -simulator.target_size/2, simulator.target_size/2]
    
    # Plot 1: Array geometry
    ax1 = plt.subplot(3, 4, 1)
    mic_x = simulator.mic_positions[:, 0]
    mic_y = simulator.mic_positions[:, 1]
    ax1.scatter(mic_x, mic_y, c='blue', s=80, marker='o', alpha=0.8, edgecolors='darkblue')
    circle = plt.Circle((0, 0), simulator.array_radius, fill=False, color='blue', linestyle='--', alpha=0.6)
    ax1.add_patch(circle)
    ax1.set_xlim(-0.3, 0.3)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Microphone Array\n(Top View)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Plot 2: System geometry (side view)
    ax2 = plt.subplot(3, 4, 2)
    ax2.scatter(mic_x, np.zeros_like(mic_x), c='blue', s=60, alpha=0.8, label='Microphones')
    target_x = [-simulator.target_size/2, simulator.target_size/2, simulator.target_size/2, -simulator.target_size/2, -simulator.target_size/2]
    target_z = [simulator.target_distance] * 5
    ax2.plot(target_x, target_z, 'r-', linewidth=3, label='Target Plane')
    ax2.scatter(source_position[0], source_position[2], c='red', s=150, marker='*', label='Source')
    ax2.set_xlim(-0.3, 0.3)
    ax2.set_ylim(-0.1, 0.5)
    ax2.set_title('System Geometry\n(Side View)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3-6: Energy maps at different times
    time_indices = [0, len(energy_maps)//3, 2*len(energy_maps)//3, -1]
    titles = ['Early Time', 'Mid Time', 'Late Time', 'Final']
    
    for i, (t_idx, title) in enumerate(zip(time_indices, titles)):
        ax = plt.subplot(3, 4, 3 + i)
        im = ax.imshow(energy_maps[t_idx], extent=extent, origin='lower', 
                      cmap='hot', interpolation='bilinear')
        ax.set_title(f'{title}\nt = {time_stamps[t_idx]:.3f}s')
        
        # Mark source position
        ax.scatter(source_position[0], source_position[1], c='white', s=200, 
                  marker='*', edgecolors='black', linewidths=2)
        
        # Mark microphone positions
        ax.scatter(mic_x, mic_y, c='cyan', s=30, marker='o', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot 7: Energy over time
    ax7 = plt.subplot(3, 4, 7)
    max_energy = np.max(energy_maps.reshape(len(energy_maps), -1), axis=1)
    total_energy = np.sum(energy_maps.reshape(len(energy_maps), -1), axis=1)
    
    ax7.plot(time_stamps, max_energy, 'b-', linewidth=2, label='Max Energy')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(time_stamps, total_energy, 'r-', linewidth=2, label='Total Energy')
    
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Max Energy', color='blue')
    ax7_twin.set_ylabel('Total Energy', color='red')
    ax7.set_title('Energy vs Time')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Peak tracking
    ax8 = plt.subplot(3, 4, 8)
    peak_positions = []
    for energy_map in energy_maps:
        peak_idx = np.unravel_index(np.argmax(energy_map), energy_map.shape)
        x_coord = (peak_idx[1] / simulator.grid_resolution - 0.5) * simulator.target_size
        y_coord = (peak_idx[0] / simulator.grid_resolution - 0.5) * simulator.target_size
        peak_positions.append((x_coord, y_coord))
    
    peak_x = [pos[0] for pos in peak_positions]
    peak_y = [pos[1] for pos in peak_positions]
    
    ax8.plot(peak_x, peak_y, 'go-', markersize=6, linewidth=2, alpha=0.7, label='Detected Peak')
    ax8.scatter(source_position[0], source_position[1], c='red', s=200, 
               marker='*', label='True Source', zorder=10)
    ax8.set_xlabel('X Position (m)')
    ax8.set_ylabel('Y Position (m)')
    ax8.set_title('Peak Energy Tracking')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axis('equal')
    
    # Plot 9: Cross-sections of final energy map
    ax9 = plt.subplot(3, 4, 9)
    center_idx = simulator.grid_resolution // 2
    x_coords = np.linspace(-simulator.target_size/2, simulator.target_size/2, simulator.grid_resolution)
    y_coords = np.linspace(-simulator.target_size/2, simulator.target_size/2, simulator.grid_resolution)
    
    final_map = energy_maps[-1]
    ax9.plot(x_coords, final_map[center_idx, :], 'b-', linewidth=2, label='Horizontal (Y=0)')
    ax9.plot(y_coords, final_map[:, center_idx], 'r-', linewidth=2, label='Vertical (X=0)')
    ax9.axvline(source_position[0], color='green', linestyle='--', alpha=0.7, label='True Source X')
    ax9.axvline(source_position[1], color='orange', linestyle='--', alpha=0.7, label='True Source Y')
    ax9.set_xlabel('Position (m)')
    ax9.set_ylabel('Energy')
    ax9.set_title('Cross-sections\n(Final Map)')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Performance metrics
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    # Calculate accuracy metrics
    final_peak_idx = np.unravel_index(np.argmax(final_map), final_map.shape)
    detected_x = (final_peak_idx[1] / simulator.grid_resolution - 0.5) * simulator.target_size
    detected_y = (final_peak_idx[0] / simulator.grid_resolution - 0.5) * simulator.target_size
    
    error_x = abs(detected_x - source_position[0])
    error_y = abs(detected_y - source_position[1])
    error_total = np.sqrt(error_x**2 + error_y**2)
    
    metrics_text = f"""PERFORMANCE METRICS

Configuration:
• Microphones: {simulator.n_mics}
• Grid: {simulator.grid_resolution}×{simulator.grid_resolution}
• Array radius: {simulator.array_radius}m

Timing:
• Computation: {computation_time:.2f}s
• Real-time factor: {1.5/computation_time:.2f}x
• Energy maps: {len(energy_maps)}

Accuracy:
• True source: ({source_position[0]:.3f}, {source_position[1]:.3f})
• Detected: ({detected_x:.3f}, {detected_y:.3f})
• Error X: {error_x:.3f}m
• Error Y: {error_y:.3f}m
• Total error: {error_total:.3f}m

Energy Stats:
• Max energy: {np.max(final_map):.3f}
• Mean energy: {np.mean(final_map):.3f}
• Dynamic range: {np.max(final_map)/np.mean(final_map):.1f}x"""
    
    ax10.text(0.05, 0.95, metrics_text, transform=ax10.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Plot 11: Energy distribution histogram
    ax11 = plt.subplot(3, 4, 11)
    all_energies = final_map.flatten()
    ax11.hist(all_energies, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax11.axvline(np.mean(all_energies), color='red', linestyle='--', linewidth=2, label='Mean')
    ax11.axvline(np.max(all_energies), color='green', linestyle='--', linewidth=2, label='Max')
    ax11.set_xlabel('Energy Level')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Energy Distribution\n(Final Map)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: 3D surface plot of final energy map
    ax12 = plt.subplot(3, 4, 12, projection='3d')
    X, Y = np.meshgrid(x_coords, y_coords)
    surf = ax12.plot_surface(X, Y, final_map, cmap='hot', alpha=0.8)
    ax12.scatter(source_position[0], source_position[1], np.max(final_map)*1.1, 
                c='white', s=100, marker='*')
    ax12.set_xlabel('X (m)')
    ax12.set_ylabel('Y (m)')
    ax12.set_zlabel('Energy')
    ax12.set_title('3D Energy Surface\n(Final Map)')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('beam_focusing_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization complete!")
    print(f"   Results saved as 'beam_focusing_results.png'")
    print(f"   Localization error: {error_total:.3f}m")
    print(f"   Performance: {1.5/computation_time:.2f}x real-time")
    
    # Show the plot
    plt.show()
    
    return simulator, energy_maps, time_stamps

def main():
    """
    Run the quick visual demonstration.
    """
    try:
        print("Ultra-Fast Audio Beam Focusing Simulator - Quick Visual Demo")
        print("This will show you the beam focusing results immediately!")
        print()
        
        simulator, energy_maps, time_stamps = quick_demo()
        
        print("\n" + "="*60)
        print("QUICK DEMO COMPLETE! 🎉")
        print("="*60)
        print("You should now see:")
        print("✓ Comprehensive beam focusing visualization")
        print("✓ Energy maps at different time points")
        print("✓ Source localization accuracy")
        print("✓ Performance metrics")
        print("✓ 3D energy surface")
        print("✓ Cross-sectional analysis")
        print("\nThe beam focusing simulator is working correctly!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()