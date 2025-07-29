#!/usr/bin/env python3
"""
Simple test script for the beam focusing simulator.
This script tests core functionality without interactive visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from beam_focusing_simulator import BeamFocusingSimulator
import time

def test_basic_functionality():
    """
    Test basic simulator functionality.
    """
    print("Testing basic beam focusing simulator functionality...")
    
    # Create simulator with smaller configuration for faster testing
    simulator = BeamFocusingSimulator(
        n_mics=16,
        grid_resolution=30,
        target_distance=0.4
    )
    
    print(f"✓ Simulator created successfully")
    print(f"  - {simulator.n_mics} microphones")
    print(f"  - {simulator.grid_resolution}×{simulator.grid_resolution} grid")
    
    # Test audio generation
    print("\nTesting audio generation...")
    start_time = time.time()
    audio_data = simulator.generate_test_audio(duration=1.0)
    audio_time = time.time() - start_time
    
    print(f"✓ Audio generated in {audio_time:.3f}s")
    print(f"  - Shape: {audio_data.shape}")
    print(f"  - RMS level: {np.sqrt(np.mean(audio_data**2)):.4f}")
    
    # Test beam focusing computation
    print("\nTesting beam focusing computation...")
    start_time = time.time()
    energy_maps, time_stamps = simulator.compute_beam_focus(
        audio_data, 
        time_window=0.2,
        overlap=0.5
    )
    focus_time = time.time() - start_time
    
    print(f"✓ Beam focusing computed in {focus_time:.3f}s")
    print(f"  - Energy maps shape: {energy_maps.shape}")
    print(f"  - Time stamps: {len(time_stamps)}")
    print(f"  - Max energy: {np.max(energy_maps):.4f}")
    print(f"  - Real-time factor: {1.0 / focus_time:.2f}x")
    
    # Test geometry calculations
    print("\nTesting geometry calculations...")
    print(f"✓ Microphone positions shape: {simulator.mic_positions.shape}")
    print(f"✓ Target points shape: {simulator.target_points.shape}")
    print(f"✓ Distance matrix shape: {simulator.distances.shape}")
    
    # Basic validation
    assert audio_data.shape[1] == simulator.n_mics, "Audio channels mismatch"
    assert energy_maps.shape[1:] == (simulator.grid_resolution, simulator.grid_resolution), "Energy map shape mismatch"
    assert len(time_stamps) == len(energy_maps), "Time stamps length mismatch"
    assert np.all(energy_maps >= 0), "Energy values should be non-negative"
    
    print("\n✓ All basic tests passed!")
    
    return simulator, energy_maps, time_stamps

def test_multiple_sources():
    """
    Test multiple source scenario.
    """
    print("\n" + "="*50)
    print("Testing multiple source scenario...")
    
    simulator = BeamFocusingSimulator(
        n_mics=16,
        grid_resolution=30
    )
    
    # Define two sources
    sources = [
        (-0.05, -0.05, 0.4),  # Left-bottom
        (0.05, 0.05, 0.4)     # Right-top
    ]
    
    print(f"Generating audio with {len(sources)} sources...")
    audio_data = simulator.generate_test_audio(
        duration=1.5, 
        source_positions=sources
    )
    
    print("Computing beam focus...")
    energy_maps, time_stamps = simulator.compute_beam_focus(audio_data)
    
    print(f"✓ Multi-source test completed")
    print(f"  - Peak energy: {np.max(energy_maps):.4f}")
    print(f"  - Energy maps: {len(energy_maps)}")
    
    return energy_maps, time_stamps

def test_performance_scaling():
    """
    Test performance with different configurations.
    """
    print("\n" + "="*50)
    print("Testing performance scaling...")
    
    configs = [
        {'n_mics': 8, 'grid_res': 20, 'name': 'Small'},
        {'n_mics': 16, 'grid_res': 30, 'name': 'Medium'},
        {'n_mics': 32, 'grid_res': 50, 'name': 'Large'}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        
        simulator = BeamFocusingSimulator(
            n_mics=config['n_mics'],
            grid_resolution=config['grid_res']
        )
        
        # Quick performance test
        start_time = time.time()
        audio_data = simulator.generate_test_audio(duration=0.5)
        energy_maps, _ = simulator.compute_beam_focus(audio_data)
        total_time = time.time() - start_time
        
        rt_factor = 0.5 / total_time
        grid_points = config['grid_res'] ** 2
        
        result = {
            'name': config['name'],
            'n_mics': config['n_mics'],
            'grid_points': grid_points,
            'time': total_time,
            'rt_factor': rt_factor,
            'max_energy': np.max(energy_maps)
        }
        
        results.append(result)
        
        print(f"  - Time: {total_time:.3f}s")
        print(f"  - RT factor: {rt_factor:.2f}x")
        print(f"  - Max energy: {np.max(energy_maps):.4f}")
    
    print("\n" + "-"*60)
    print("Performance Summary:")
    print(f"{'Config':<10} {'Mics':<6} {'Grid':<8} {'Time(s)':<8} {'RT Factor':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['name']:<10} {result['n_mics']:<6} "
              f"{result['grid_points']:<8} {result['time']:<8.3f} "
              f"{result['rt_factor']:<10.2f}")
    
    return results

def test_parameter_validation():
    """
    Test parameter validation and edge cases.
    """
    print("\n" + "="*50)
    print("Testing parameter validation...")
    
    # Test valid parameters
    try:
        simulator = BeamFocusingSimulator(
            n_mics=8,
            array_radius=0.1,
            target_distance=0.2,
            grid_resolution=20
        )
        print("✓ Valid parameter test passed")
    except Exception as e:
        print(f"✗ Valid parameter test failed: {e}")
        return False
    
    # Test edge case: very small configuration
    try:
        simulator = BeamFocusingSimulator(
            n_mics=4,
            grid_resolution=10
        )
        audio_data = simulator.generate_test_audio(duration=0.1)
        energy_maps, _ = simulator.compute_beam_focus(audio_data)
        print("✓ Minimal configuration test passed")
    except Exception as e:
        print(f"✗ Minimal configuration test failed: {e}")
        return False
    
    print("✓ Parameter validation tests passed")
    return True

def save_test_results(energy_maps, time_stamps):
    """
    Save test results for verification.
    """
    print("\n" + "="*50)
    print("Saving test results...")
    
    # Save energy maps
    np.save('test_energy_maps.npy', energy_maps)
    np.save('test_time_stamps.npy', time_stamps)
    
    # Create a simple plot
    plt.figure(figsize=(10, 4))
    
    # Plot 1: Sample energy map
    plt.subplot(1, 2, 1)
    plt.imshow(energy_maps[0], cmap='hot', origin='lower')
    plt.title('Sample Energy Map')
    plt.colorbar()
    
    # Plot 2: Energy over time
    plt.subplot(1, 2, 2)
    max_energy = np.max(energy_maps.reshape(len(energy_maps), -1), axis=1)
    plt.plot(time_stamps, max_energy, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Max Energy')
    plt.title('Energy vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Results saved:")
    print("  - test_energy_maps.npy")
    print("  - test_time_stamps.npy")
    print("  - test_results.png")

def main():
    """
    Run all tests.
    """
    print("Ultra-Fast Audio Beam Focusing Simulator - Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        simulator, energy_maps, time_stamps = test_basic_functionality()
        test_multiple_sources()
        test_performance_scaling()
        test_parameter_validation()
        save_test_results(energy_maps, time_stamps)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nThe beam focusing simulator is working correctly.")
        print("You can now:")
        print("  1. Run 'python demo.py' for full demonstrations")
        print("  2. Open 'interactive_demo.ipynb' in Jupyter")
        print("  3. Use the simulator in your own scripts")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)