"""Audio Focusing Simulation 메인 실행 파일"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import argparse
import time

# 로컬 모듈 import
from mic_array import CircularMicArray
from signal_model import AudioSource, SignalSimulator, SignalType
from fd_beamforming import FDBeamformer, BeamformingType, create_target_grid
from plot_utils import AudioVisualization, save_results_to_hdf5

class AudioFocusingSimulator:
    """오디오 포커싱 시뮬레이터 메인 클래스"""
    
    def __init__(self, config: dict):
        """
        시뮬레이터 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        
        # 마이크 어레이 설정
        self.mic_array = CircularMicArray(
            num_mics=config['num_mics'],
            radius=config['array_radius'],
            center=(0.0, 0.0)
        )
        
        # 신호 시뮬레이터 설정
        self.signal_simulator = SignalSimulator(
            sound_speed=config['sound_speed']
        )
        
        # Beamforming 타입 설정
        beamforming_type_map = {
            'filter_and_sum': BeamformingType.FILTER_AND_SUM,
            'mvdr': BeamformingType.MVDR,
            'robust_mvdr': BeamformingType.ROBUST_MVDR
        }
        
        beamforming_type = beamforming_type_map.get(
            config.get('beamforming_type', 'filter_and_sum'),
            BeamformingType.FILTER_AND_SUM
        )
        
        # Beamformer 설정
        self.beamformer = FDBeamformer(
            sample_rate=config['sample_rate'],
            fft_size=config['fft_size'],
            hop_size=config['hop_size'],
            sound_speed=config['sound_speed'],
            beamforming_type=beamforming_type,
            diagonal_loading=config.get('diagonal_loading', 1e-6),
            regularization=config.get('regularization', 1e-3)
        )
        
        # 시각화 설정
        self.visualizer = AudioVisualization()
        
        # 타겟 그리드 생성
        self.target_grid = create_target_grid(
            grid_size=config['grid_size'],
            physical_size=config['physical_size'],
            z_distance=config['z_distance']
        )
        
    def add_audio_source(self, position: Tuple[float, float, float], 
                        signal_type: SignalType = SignalType.SINE_WAVE,
                        frequency: float = 1000.0, amplitude: float = 1.0):
        """
        음원 추가
        
        Args:
            position: 음원 위치 (x, y, z)
            signal_type: 신호 타입
            frequency: 주파수
            amplitude: 진폭
        """
        source = AudioSource(position, signal_type, frequency, amplitude)
        self.signal_simulator.add_source(source)
        print(f"Added audio source at {position} with {signal_type.value} signal")
    
    def run_simulation(self, duration: float = 1.0, show_plots: bool = True, 
                      save_results: bool = False, output_dir: str = "./") -> dict:
        """
        시뮬레이션 실행
        
        Args:
            duration: 신호 길이 (초)
            show_plots: 플롯 표시 여부
            save_results: 결과 저장 여부
            output_dir: 출력 디렉토리
            
        Returns:
            results: 결과 딕셔너리
        """
        print("Starting audio focusing simulation...")
        start_time = time.time()
        
        # 1. 마이크 신호 시뮬레이션
        print("1. Simulating microphone signals...")
        mic_positions = self.mic_array.get_mic_positions()
        mic_signals = self.signal_simulator.simulate_mic_signals(
            mic_positions=mic_positions,
            duration=duration,
            sample_rate=self.config['sample_rate'],
            add_noise=self.config['add_noise'],
            noise_level=self.config['noise_level']
        )
        
        # 2. Beamforming 수행
        print("2. Performing frequency domain beamforming...")
        energy_map = self.beamformer.process_signals(
            mic_signals=mic_signals,
            mic_positions=mic_positions,
            target_grid=self.target_grid,
            grid_shape=self.config['grid_size']
        )
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # 결과 딕셔너리 생성
        results = {
            'energy_map': energy_map,
            'mic_signals': mic_signals,
            'mic_positions': mic_positions,
            'target_grid': self.target_grid,
            'processing_time': processing_time,
            'config': self.config
        }
        
        # 3. 시각화
        if show_plots:
            print("3. Generating visualizations...")
            self._create_visualizations(results, output_dir, save_results)
        
        # 4. 결과 저장
        if save_results:
            print("4. Saving results...")
            self._save_results(results, output_dir)
        
        return results
    
    def _create_visualizations(self, results: dict, output_dir: str, save_plots: bool):
        """시각화 생성"""
        # 통합 결과 시각화
        fig_main = self.visualizer.plot_interactive_results(
            energy_map=results['energy_map'],
            mic_signals=results['mic_signals'],
            mic_positions=results['mic_positions'],
            sample_rate=self.config['sample_rate'],
            physical_size=self.config['physical_size']
        )
        
        if save_plots:
            fig_main.savefig(f"{output_dir}/simulation_results.png", dpi=300, bbox_inches='tight')
        
        # 에너지 맵 상세 시각화
        fig_energy = self.visualizer.plot_energy_map(
            energy_map=results['energy_map'],
            physical_size=self.config['physical_size'],
            title="Audio Focusing Energy Map - Detailed View"
        )
        
        if save_plots:
            fig_energy.savefig(f"{output_dir}/energy_map.png", dpi=300, bbox_inches='tight')
        
        # 마이크 어레이 레이아웃
        fig_layout = self.visualizer.plot_mic_array_layout(
            mic_positions=results['mic_positions'],
            array_radius=self.config['array_radius'],
            target_grid_size=self.config['physical_size'],
            z_distance=self.config['z_distance']
        )
        
        if save_plots:
            fig_layout.savefig(f"{output_dir}/mic_array_layout.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _save_results(self, results: dict, output_dir: str):
        """결과 저장"""
        # HDF5 형식으로 저장
        metadata = {
            'num_mics': self.config['num_mics'],
            'array_radius': self.config['array_radius'],
            'sample_rate': self.config['sample_rate'],
            'fft_size': self.config['fft_size'],
            'grid_size_x': self.config['grid_size'][0],
            'grid_size_y': self.config['grid_size'][1],
            'physical_size_x': self.config['physical_size'][0],
            'physical_size_y': self.config['physical_size'][1],
            'z_distance': self.config['z_distance'],
            'processing_time': results['processing_time']
        }
        
        save_results_to_hdf5(
            filename=f"{output_dir}/simulation_results.h5",
            energy_map=results['energy_map'],
            mic_signals=results['mic_signals'],
            mic_positions=results['mic_positions'],
            metadata=metadata
        )

def create_default_config() -> dict:
    """기본 설정 생성"""
    return {
        # 마이크 어레이 설정
        'num_mics': 8,
        'array_radius': 0.1,  # m
        
        # 신호 설정
        'sample_rate': 44100,  # Hz
        'sound_speed': 343.0,  # m/s
        'add_noise': True,
        'noise_level': 0.01,
        
        # Beamforming 설정
        'fft_size': 1024,
        'hop_size': 512,
        'beamforming_type': 'filter_and_sum',  # 'filter_and_sum', 'mvdr', 'robust_mvdr'
        'diagonal_loading': 1e-6,
        'regularization': 1e-3,
        
        # 타겟 그리드 설정
        'grid_size': (70, 70),
        'physical_size': (1.0, 1.0),  # m
        'z_distance': 1.0,  # m
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Audio Focusing Simulation')
    parser.add_argument('--num-mics', type=int, default=8, help='Number of microphones')
    parser.add_argument('--array-radius', type=float, default=0.1, help='Array radius (m)')
    parser.add_argument('--duration', type=float, default=1.0, help='Signal duration (s)')
    parser.add_argument('--frequency', type=float, default=1000.0, help='Source frequency (Hz)')
    parser.add_argument('--source-x', type=float, default=0.2, help='Source X position (m)')
    parser.add_argument('--source-y', type=float, default=0.3, help='Source Y position (m)')
    parser.add_argument('--source-z', type=float, default=1.0, help='Source Z position (m)')
    parser.add_argument('--beamforming-type', type=str, default='filter_and_sum',
                        choices=['filter_and_sum', 'mvdr', 'robust_mvdr'],
                        help='Beamforming algorithm type')
    parser.add_argument('--diagonal-loading', type=float, default=1e-6,
                        help='Diagonal loading value for MVDR')
    parser.add_argument('--regularization', type=float, default=1e-3,
                        help='Regularization parameter')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot display')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
    parser.add_argument('--output-dir', type=str, default='./', help='Output directory')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = create_default_config()
    config['num_mics'] = args.num_mics
    config['array_radius'] = args.array_radius
    config['beamforming_type'] = args.beamforming_type
    config['diagonal_loading'] = args.diagonal_loading
    config['regularization'] = args.regularization
    
    # 시뮬레이터 생성
    simulator = AudioFocusingSimulator(config)
    
    # 음원 추가
    simulator.add_audio_source(
        position=(args.source_x, args.source_y, args.source_z),
        signal_type=SignalType.SINE_WAVE,
        frequency=args.frequency,
        amplitude=1.0
    )
    
    # 시뮬레이션 실행
    results = simulator.run_simulation(
        duration=args.duration,
        show_plots=not args.no_plots,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    # 결과 요약 출력
    print("\n=== Simulation Summary ===")
    print(f"Number of microphones: {config['num_mics']}")
    print(f"Array radius: {config['array_radius']:.3f} m")
    print(f"Grid size: {config['grid_size']}")
    print(f"Physical size: {config['physical_size']} m")
    print(f"Processing time: {results['processing_time']:.2f} s")
    print(f"Max energy: {np.max(results['energy_map']):.2e}")
    print(f"Energy map shape: {results['energy_map'].shape}")

if __name__ == "__main__":
    main()