"""시각화 유틸리티 모듈"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from typing import Optional, Tuple, List
import warnings

class AudioVisualization:
    """오디오 포커싱 시각화 클래스"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        시각화 클래스 초기화
        
        Args:
            figsize: Figure 크기
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_energy_map(self, energy_map: np.ndarray, 
                       physical_size: Tuple[float, float] = (1.0, 1.0),
                       title: str = "Audio Focusing Energy Map",
                       colormap: str = 'hot',
                       log_scale: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        에너지 맵 시각화
        
        Args:
            energy_map: 에너지 맵 (height, width)
            physical_size: 물리적 크기 (height_m, width_m)
            title: 제목
            colormap: 컬러맵
            log_scale: 로그 스케일 사용 여부
            save_path: 저장 경로
            
        Returns:
            figure: matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        height_m, width_m = physical_size
        extent = [-width_m/2, width_m/2, -height_m/2, height_m/2]
        
        # 로그 스케일 적용
        if log_scale:
            # 0 값 처리를 위해 작은 값 추가
            energy_map_plot = energy_map + np.finfo(float).eps
            norm = LogNorm(vmin=energy_map_plot.min(), vmax=energy_map_plot.max())
        else:
            energy_map_plot = energy_map
            norm = None
        
        # 히트맵 그리기
        im = ax.imshow(energy_map_plot, extent=extent, origin='lower', 
                      cmap=colormap, norm=norm, aspect='equal')
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Energy Level' + (' (log scale)' if log_scale else ''), rotation=270, labelpad=20)
        
        # 축 설정
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_mic_array_layout(self, mic_positions: np.ndarray, 
                             array_radius: float,
                             target_grid_size: Tuple[float, float] = (1.0, 1.0),
                             z_distance: float = 1.0,
                             title: str = "Microphone Array Layout") -> plt.Figure:
        """
        마이크 어레이 레이아웃 시각화
        
        Args:
            mic_positions: 마이크 위치 (N_mics, 2)
            array_radius: 어레이 반지름
            target_grid_size: 타겟 그리드 크기
            z_distance: Z 거리
            title: 제목
            
        Returns:
            figure: matplotlib Figure 객체
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D Top view
        ax1.scatter(mic_positions[:, 0], mic_positions[:, 1], 
                   c='red', s=100, marker='o', label='Microphones')
        
        # 마이크 번호 표시
        for i, (x, y) in enumerate(mic_positions):
            ax1.annotate(f'M{i+1}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # 원형 어레이 경계 표시
        circle = patches.Circle((0, 0), array_radius, fill=False, 
                               linestyle='--', color='red', alpha=0.5)
        ax1.add_patch(circle)
        
        # 타겟 그리드 영역 표시
        width, height = target_grid_size
        rect = patches.Rectangle((-width/2, -height/2), width, height,
                               fill=False, linestyle='-', color='blue', alpha=0.7)
        ax1.add_patch(rect)
        
        ax1.set_xlim(-max(array_radius, width/2) * 1.2, max(array_radius, width/2) * 1.2)
        ax1.set_ylim(-max(array_radius, height/2) * 1.2, max(array_radius, height/2) * 1.2)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Top View')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        
        # 3D Side view (XZ plane)
        ax2.scatter(mic_positions[:, 0], np.zeros(len(mic_positions)), 
                   c='red', s=100, marker='o', label='Microphones')
        
        # 타겟 평면 표시
        ax2.plot([-width/2, width/2], [z_distance, z_distance], 
                'b-', linewidth=3, label='Target Plane')
        
        ax2.set_xlim(-max(array_radius, width/2) * 1.2, max(array_radius, width/2) * 1.2)
        ax2.set_ylim(-0.1, z_distance * 1.2)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('Side View (XZ plane)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_signal_waveforms(self, signals: np.ndarray, sample_rate: int,
                             mic_indices: Optional[List[int]] = None,
                             title: str = "Microphone Signals") -> plt.Figure:
        """
        마이크 신호 파형 시각화
        
        Args:
            signals: 신호 (N_mics, N_samples)
            sample_rate: 샘플링 레이트
            mic_indices: 표시할 마이크 인덱스 (None이면 모두 표시)
            title: 제목
            
        Returns:
            figure: matplotlib Figure 객체
        """
        n_mics, n_samples = signals.shape
        time_axis = np.arange(n_samples) / sample_rate
        
        if mic_indices is None:
            mic_indices = list(range(min(n_mics, 8)))  # 최대 8개만 표시
        
        n_plots = len(mic_indices)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots), sharex=True)
        
        if n_plots == 1:
            axes = [axes]
        
        for i, mic_idx in enumerate(mic_indices):
            axes[i].plot(time_axis, signals[mic_idx], 'b-', linewidth=0.8)
            axes[i].set_ylabel(f'Mic {mic_idx+1}\nAmplitude')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, time_axis[-1])
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_frequency_spectrum(self, signals: np.ndarray, sample_rate: int,
                               mic_index: int = 0,
                               title: str = "Frequency Spectrum") -> plt.Figure:
        """
        주파수 스펙트럼 시각화
        
        Args:
            signals: 신호 (N_mics, N_samples)
            sample_rate: 샘플링 레이트
            mic_index: 분석할 마이크 인덱스
            title: 제목
            
        Returns:
            figure: matplotlib Figure 객체
        """
        signal = signals[mic_index]
        n_samples = len(signal)
        
        # FFT 계산
        fft_result = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(n_samples, 1/sample_rate)
        magnitude = np.abs(fft_result)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 선형 스케일
        ax1.plot(frequencies, magnitude, 'b-', linewidth=0.8)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'{title} - Linear Scale (Mic {mic_index+1})')
        ax1.grid(True, alpha=0.3)
        
        # 로그 스케일
        ax2.semilogx(frequencies[1:], 20*np.log10(magnitude[1:] + np.finfo(float).eps), 'r-', linewidth=0.8)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title(f'{title} - Log Scale (Mic {mic_index+1})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_results(self, energy_map: np.ndarray, 
                               mic_signals: np.ndarray,
                               mic_positions: np.ndarray,
                               sample_rate: int,
                               physical_size: Tuple[float, float] = (1.0, 1.0)) -> plt.Figure:
        """
        통합 결과 시각화
        
        Args:
            energy_map: 에너지 맵
            mic_signals: 마이크 신호
            mic_positions: 마이크 위치
            sample_rate: 샘플링 레이트
            physical_size: 물리적 크기
            
        Returns:
            figure: matplotlib Figure 객체
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 에너지 맵 (상단 좌측)
        ax1 = plt.subplot(2, 2, 1)
        height_m, width_m = physical_size
        extent = [-width_m/2, width_m/2, -height_m/2, height_m/2]
        
        energy_plot = energy_map + np.finfo(float).eps
        im = ax1.imshow(energy_plot, extent=extent, origin='lower', 
                       cmap='hot', norm=LogNorm(), aspect='equal')
        plt.colorbar(im, ax=ax1, shrink=0.8)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Audio Focusing Energy Map')
        ax1.grid(True, alpha=0.3)
        
        # 마이크 어레이 레이아웃 (상단 우측)
        ax2 = plt.subplot(2, 2, 2)
        ax2.scatter(mic_positions[:, 0], mic_positions[:, 1], 
                   c='red', s=100, marker='o')
        for i, (x, y) in enumerate(mic_positions):
            ax2.annotate(f'M{i+1}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Microphone Array Layout')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 신호 파형 (하단 좌측)
        ax3 = plt.subplot(2, 2, 3)
        time_axis = np.arange(mic_signals.shape[1]) / sample_rate
        ax3.plot(time_axis, mic_signals[0], 'b-', linewidth=0.8, label='Mic 1')
        if mic_signals.shape[0] > 1:
            ax3.plot(time_axis, mic_signals[1], 'r-', linewidth=0.8, label='Mic 2')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Sample Microphone Signals')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 주파수 스펙트럼 (하단 우측)
        ax4 = plt.subplot(2, 2, 4)
        signal = mic_signals[0]
        fft_result = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(fft_result)
        
        ax4.semilogx(frequencies[1:], 20*np.log10(magnitude[1:] + np.finfo(float).eps), 
                    'g-', linewidth=0.8)
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude (dB)')
        ax4.set_title('Frequency Spectrum (Mic 1)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Audio Focusing Simulation Results', fontsize=16)
        plt.tight_layout()
        
        return fig

def save_results_to_hdf5(filename: str, energy_map: np.ndarray, 
                        mic_signals: np.ndarray, mic_positions: np.ndarray,
                        metadata: dict):
    """
    결과를 HDF5 파일로 저장
    
    Args:
        filename: 저장할 파일명
        energy_map: 에너지 맵
        mic_signals: 마이크 신호
        mic_positions: 마이크 위치
        metadata: 메타데이터
    """
    try:
        import h5py
        
        with h5py.File(filename, 'w') as f:
            # 데이터 저장
            f.create_dataset('energy_map', data=energy_map)
            f.create_dataset('mic_signals', data=mic_signals)
            f.create_dataset('mic_positions', data=mic_positions)
            
            # 메타데이터 저장
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                meta_group.attrs[key] = value
                
        print(f"Results saved to {filename}")
        
    except ImportError:
        warnings.warn("h5py not available. Results not saved.")
    except Exception as e:
        warnings.warn(f"Error saving results: {e}")