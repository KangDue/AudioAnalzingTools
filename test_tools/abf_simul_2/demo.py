#!/usr/bin/env python3
"""
Demo script for the Ultra-Fast Audio Beam Focusing Simulator

This script demonstrates various capabilities of the beam focusing simulator:
1. Basic beam focusing with single source
2. Multiple source scenarios
3. Performance benchmarking
4. Parameter sensitivity analysis
5. Visualization options

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from beam_focusing_simulator import BeamFocusingSimulator
import time


def demo_basic_focusing():
    """
    Demonstrate basic beam focusing with a single source.
    """
    print("=" * 60)
    print("DEMO 1: Basic Beam Focusing")
    print("=" * 60)
    
    # Create simulator with default parameters
    simulator = BeamFocusingSimulator()
    
    # Generate test audio (single source at center)
    print("\nGenerating test audio with single source at center...")
    audio_data = simulator.generate_test_audio(duration=3.0)
    
    # Compute beam focusing
    print("Computing beam focus energy maps...")
    energy_maps, time_stamps = simulator.compute_beam_focus(audio_data, time_window=0.2)
    
    # Show statistics
    print(f"\nResults:")
    print(f"  Generated {len(energy_maps)} energy maps")
    print(f"  Time range: {time_stamps[0]:.2f}s to {time_stamps[-1]:.2f}s")
    print(f"  Max energy: {np.max(energy_maps):.4f}")
    print(f"  Energy map shape: {energy_maps[0].shape}")
    
    # Visualize
    print("\nCreating visualization...")
    anim = simulator.visualize_energy_maps(energy_maps, time_stamps)
    
    return simulator, energy_maps, time_stamps


def demo_multiple_sources():
    """
    Demonstrate beam focusing with multiple sources.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Multiple Source Beam Focusing")
    print("=" * 60)
    
    # Create simulator
    simulator = BeamFocusingSimulator()
    
    # Define multiple source positions
    source_positions = [
        (-0.1, -0.1, 0.4),  # Left-bottom
        (0.1, 0.1, 0.4),   # Right-top
        (0.0, 0.0, 0.4)    # Center
    ]
    
    print(f"\nGenerating audio with {len(source_positions)} sources:")
    for i, pos in enumerate(source_positions):
        print(f"  Source {i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})m")
    
    # Generate test audio
    audio_data = simulator.generate_test_audio(duration=4.0, source_positions=source_positions)
    
    # Compute beam focusing
    print("\nComputing beam focus for multiple sources...")
    energy_maps, time_stamps = simulator.compute_beam_focus(audio_data, time_window=0.15)
    
    # Analyze results
    print(f"\nMultiple source results:")
    print(f"  Energy maps: {len(energy_maps)}")
    print(f"  Peak energy: {np.max(energy_maps):.4f}")
    
    # Find energy peaks in final map
    final_map = energy_maps[-1]
    peak_indices = np.unravel_index(np.argmax(final_map), final_map.shape)
    print(f"  Peak location (grid): {peak_indices}")
    
    # Visualize
    print("\nCreating multi-source visualization...")
    anim = simulator.visualize_energy_maps(energy_maps, time_stamps)
    
    return simulator, energy_maps, time_stamps


def demo_parameter_sensitivity():
    """
    Demonstrate sensitivity to different parameters.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # Test different grid resolutions
    resolutions = [30, 50, 70, 90]
    performance_results = []
    
    print("\nTesting different grid resolutions:")
    
    for resolution in resolutions:
        print(f"\n  Testing {resolution}×{resolution} grid...")
        
        simulator = BeamFocusingSimulator(grid_resolution=resolution)
        
        # Quick performance test
        start_time = time.time()
        audio_data = simulator.generate_test_audio(duration=1.0)
        energy_maps, _ = simulator.compute_beam_focus(audio_data, time_window=0.2)
        total_time = time.time() - start_time
        
        performance_results.append({
            'resolution': resolution,
            'grid_points': resolution**2,
            'time': total_time,
            'max_energy': np.max(energy_maps)
        })
        
        print(f"    Time: {total_time:.2f}s, Max energy: {np.max(energy_maps):.4f}")
    
    # Plot performance vs resolution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    resolutions_list = [r['resolution'] for r in performance_results]
    times_list = [r['time'] for r in performance_results]
    plt.plot(resolutions_list, times_list, 'bo-')
    plt.xlabel('Grid Resolution')
    plt.ylabel('Computation Time (s)')
    plt.title('Performance vs Resolution')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    grid_points_list = [r['grid_points'] for r in performance_results]
    plt.plot(grid_points_list, times_list, 'ro-')
    plt.xlabel('Total Grid Points')
    plt.ylabel('Computation Time (s)')
    plt.title('Performance vs Grid Points')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    energies_list = [r['max_energy'] for r in performance_results]
    plt.plot(resolutions_list, energies_list, 'go-')
    plt.xlabel('Grid Resolution')
    plt.ylabel('Max Energy')
    plt.title('Energy vs Resolution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return performance_results


def demo_performance_benchmark():
    """
    Comprehensive performance benchmark.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Performance Benchmark")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {'name': 'Small (16 mics, 50×50)', 'n_mics': 16, 'grid_resolution': 50},
        {'name': 'Medium (32 mics, 70×70)', 'n_mics': 32, 'grid_resolution': 70},
        {'name': 'Large (64 mics, 100×100)', 'n_mics': 64, 'grid_resolution': 100},
    ]
    
    benchmark_results = []
    
    for config in configs:
        print(f"\nBenchmarking: {config['name']}")
        
        simulator = BeamFocusingSimulator(
            n_mics=config['n_mics'],
            grid_resolution=config['grid_resolution']
        )
        
        # Run benchmark
        results = simulator.benchmark_performance(duration=2.0)
        results['config'] = config['name']
        results['n_mics'] = config['n_mics']
        results['grid_resolution'] = config['grid_resolution']
        
        benchmark_results.append(results)
    
    # Summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Time (s)':<10} {'RT Factor':<12} {'Samples/s':<12}")
    print("-" * 80)
    
    for result in benchmark_results:
        print(f"{result['config']:<25} {result['beam_focus_time']:<10.3f} "
              f"{result['real_time_factor']:<12.2f} {result['samples_per_second']:<12.0f}")
    
    return benchmark_results


def demo_visualization_options():
    """
    Demonstrate different visualization options.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Visualization Options")
    print("=" * 60)
    
    # Create simulator
    simulator = BeamFocusingSimulator()
    
    # Generate interesting test case (moving source simulation)
    print("\nGenerating dynamic source scenario...")
    
    # Simulate moving source by using different positions over time
    duration = 5.0
    n_segments = 5
    segment_duration = duration / n_segments
    
    all_audio = []
    
    for i in range(n_segments):
        # Source moves in a circle
        angle = 2 * np.pi * i / n_segments
        x = 0.1 * np.cos(angle)
        y = 0.1 * np.sin(angle)
        source_pos = [(x, y, 0.4)]
        
        segment_audio = simulator.generate_test_audio(
            duration=segment_duration, 
            source_positions=source_pos
        )
        all_audio.append(segment_audio)
    
    # Concatenate all segments
    audio_data = np.vstack(all_audio)
    
    # Compute beam focusing with high temporal resolution
    print("Computing high-resolution beam focus...")
    energy_maps, time_stamps = simulator.compute_beam_focus(
        audio_data, 
        time_window=0.1, 
        overlap=0.8
    )
    
    print(f"Generated {len(energy_maps)} energy maps for visualization")
    
    # Create visualization
    print("\nCreating enhanced visualization...")
    anim = simulator.visualize_energy_maps(energy_maps, time_stamps, 
                                         save_animation=False)
    
    # Additional analysis plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Energy over time at center
    plt.subplot(1, 3, 1)
    center_idx = simulator.grid_resolution // 2
    center_energy = energy_maps[:, center_idx, center_idx]
    plt.plot(time_stamps, center_energy, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy at Center')
    plt.title('Center Point Energy vs Time')
    plt.grid(True)
    
    # Plot 2: Total energy over time
    plt.subplot(1, 3, 2)
    total_energy = np.sum(energy_maps.reshape(len(energy_maps), -1), axis=1)
    plt.plot(time_stamps, total_energy, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy')
    plt.title('Total Energy vs Time')
    plt.grid(True)
    
    # Plot 3: Peak location over time
    plt.subplot(1, 3, 3)
    peak_x = []
    peak_y = []
    
    for energy_map in energy_maps:
        peak_idx = np.unravel_index(np.argmax(energy_map), energy_map.shape)
        # Convert grid indices to physical coordinates
        x_coord = (peak_idx[1] / simulator.grid_resolution - 0.5) * simulator.target_size
        y_coord = (peak_idx[0] / simulator.grid_resolution - 0.5) * simulator.target_size
        peak_x.append(x_coord)
        peak_y.append(y_coord)
    
    plt.plot(peak_x, peak_y, 'go-', markersize=4, linewidth=1)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Peak Energy Location Track')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return energy_maps, time_stamps


def main():
    """
    Run all demonstrations.
    """
    print("Ultra-Fast Audio Beam Focusing Simulator - Demo Suite")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_basic_focusing()
        demo_multiple_sources()
        demo_parameter_sensitivity()
        demo_performance_benchmark()
        demo_visualization_options()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()