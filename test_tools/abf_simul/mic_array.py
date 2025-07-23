"""마이크 어레이 설정 및 steering vector 계산 모듈"""

import numpy as np
from typing import Tuple, List

class CircularMicArray:
    """원형 마이크 어레이 클래스"""
    
    def __init__(self, num_mics: int = 8, radius: float = 0.1, center: Tuple[float, float] = (0.0, 0.0)):
        """
        원형 마이크 어레이 초기화
        
        Args:
            num_mics: 마이크 개수 (기본값: 8)
            radius: 원형 배열 반지름 (m, 기본값: 0.1)
            center: 배열 중심 좌표 (x, y)
        """
        self.num_mics = num_mics
        self.radius = radius
        self.center = center
        self.mic_positions = self._calculate_mic_positions()
        
    def _calculate_mic_positions(self) -> np.ndarray:
        """마이크 위치 계산 (2D 좌표)"""
        angles = np.linspace(0, 2*np.pi, self.num_mics, endpoint=False)
        x_positions = self.center[0] + self.radius * np.cos(angles)
        y_positions = self.center[1] + self.radius * np.sin(angles)
        return np.column_stack((x_positions, y_positions))
    
    def calculate_steering_vectors(self, target_grid: np.ndarray, frequency: float, 
                                 sound_speed: float = 343.0) -> np.ndarray:
        """
        타겟 그리드에 대한 steering vector 계산
        
        Args:
            target_grid: 타겟 포인트들의 3D 좌표 (N, 3)
            frequency: 주파수 (Hz)
            sound_speed: 음속 (m/s)
            
        Returns:
            steering_vectors: 복소수 steering vector (N_points, N_mics)
        """
        n_points = len(target_grid)
        n_mics = len(self.mic_positions)
        
        # 3D 마이크 위치 (z=0 가정)
        mic_pos_3d = np.column_stack([self.mic_positions, np.zeros(n_mics)])
        
        # 거리 계산 (N_points, N_mics)
        distances = self.calculate_distances(target_grid)
        
        # Steering vector 계산
        if frequency == 0:  # DC 성분 처리
            steering_vectors = np.ones((n_points, n_mics), dtype=complex)
        else:
            # 위상 보정 인자 계산 (최신 MVDR 기반)
            phase_correction = -2j * np.pi * frequency * distances / sound_speed
            steering_vectors = np.exp(phase_correction)
        
        return steering_vectors
    
    def calculate_distances(self, target_points: np.ndarray) -> np.ndarray:
        """
        각 마이크에서 타겟 포인트까지의 거리 계산
        
        Args:
            target_points: 타겟 포인트들의 3D 좌표 (N, 3)
            
        Returns:
            distances: 거리 행렬 (N_points, N_mics)
        """
        n_mics = len(self.mic_positions)
        
        # 3D 마이크 위치 (z=0 가정)
        mic_pos_3d = np.column_stack([self.mic_positions, np.zeros(n_mics)])
        
        # 거리 계산: ||target_point - mic_position||
        # Broadcasting을 사용하여 효율적으로 계산
        distances = np.linalg.norm(
            target_points[:, np.newaxis, :] - mic_pos_3d[np.newaxis, :, :], 
            axis=2
        )
        
        return distances
    
    def get_mic_positions(self) -> np.ndarray:
        """마이크 위치 반환"""
        return self.mic_positions
    
    def get_array_info(self) -> dict:
        """어레이 정보 반환"""
        return {
            'num_mics': self.num_mics,
            'radius': self.radius,
            'center': self.center,
            'mic_positions': self.mic_positions,
            'array_type': 'circular'
        }
    
    def calculate_array_manifold(self, frequencies: np.ndarray, target_points: np.ndarray,
                               sound_speed: float = 343.0) -> np.ndarray:
        """
        다중 주파수에 대한 array manifold 계산 (최신 MVDR/LCMV 기법 적용)
        
        Args:
            frequencies: 주파수 배열 (N_freq,)
            target_points: 타겟 포인트들 (N_points, 3)
            sound_speed: 음속 (m/s)
            
        Returns:
            manifold: Array manifold (N_freq, N_points, N_mics)
        """
        n_freq = len(frequencies)
        n_points = len(target_points)
        n_mics = self.num_mics
        
        manifold = np.zeros((n_freq, n_points, n_mics), dtype=complex)
        
        # 거리 계산 (한 번만 수행)
        distances = self.calculate_distances(target_points)
        
        # 각 주파수에 대해 steering vector 계산
        for freq_idx, freq in enumerate(frequencies):
            manifold[freq_idx] = self.calculate_steering_vectors(target_points, freq, sound_speed)
        
        return manifold
    
    def plot_array_layout(self, title: str = "Circular Microphone Array Layout"):
        """마이크 어레이 레이아웃 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 마이크 위치 표시
        ax.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                  c='red', s=100, marker='o', label='Microphones')
        
        # 마이크 번호 표시
        for i, (x, y) in enumerate(self.mic_positions):
            ax.annotate(f'M{i+1}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # 원형 어레이 경계 표시
        circle = patches.Circle(self.center, self.radius, fill=False, 
                               linestyle='--', color='red', alpha=0.7, linewidth=2)
        ax.add_patch(circle)
        
        # 중심점 표시
        ax.plot(self.center[0], self.center[1], 'ko', markersize=8, label='Array Center')
        
        # 축 설정
        ax.set_xlim(-self.radius * 1.5, self.radius * 1.5)
        ax.set_ylim(-self.radius * 1.5, self.radius * 1.5)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        return fig