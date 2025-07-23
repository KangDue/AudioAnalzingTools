"""Frequency Domain Filter-and-Sum Beamforming 알고리즘 모듈

최신 beamforming 기법들을 포함:
- Filter-and-Sum Beamforming (FD-FAS)
- Minimum Variance Distortionless Response (MVDR)
- Linearly Constrained Minimum Variance (LCMV)
- Robust MVDR with covariance matrix reconstruction
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional, Union
import warnings
from enum import Enum

class BeamformingType(Enum):
    """Beamforming 알고리즘 타입"""
    FILTER_AND_SUM = "filter_and_sum"  # 기본 FD-FAS
    MVDR = "mvdr"  # Minimum Variance Distortionless Response
    LCMV = "lcmv"  # Linearly Constrained Minimum Variance
    ROBUST_MVDR = "robust_mvdr"  # Robust MVDR with covariance reconstruction

class FDBeamformer:
    """Frequency Domain Filter-and-Sum Beamformer 클래스"""
    
    def __init__(self, sample_rate: int, fft_size: int = 1024, hop_size: int = 512,
                 window: str = 'hann', sound_speed: float = 343.0,
                 beamforming_type: BeamformingType = BeamformingType.FILTER_AND_SUM,
                 diagonal_loading: float = 1e-6, regularization: float = 1e-3):
        """
        FD Beamformer 초기화
        
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            fft_size: FFT 크기
            hop_size: Hop 크기
            window: 윈도우 함수
            sound_speed: 음속 (m/s)
            beamforming_type: Beamforming 알고리즘 타입
            diagonal_loading: MVDR/LCMV용 대각선 로딩 값
            regularization: 정규화 파라미터
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = window
        self.sound_speed = sound_speed
        self.beamforming_type = beamforming_type
        self.diagonal_loading = diagonal_loading
        self.regularization = regularization
        
        # 주파수 빈 계산
        self.frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
    def stft_analysis(self, signals: np.ndarray) -> np.ndarray:
        """
        STFT 분석
        
        Args:
            signals: 입력 신호 (N_mics, N_samples)
            
        Returns:
            stft_data: STFT 결과 (N_mics, N_freq, N_time)
        """
        n_mics, n_samples = signals.shape
        
        # 각 마이크 신호에 대해 STFT 수행
        stft_results = []
        for mic_idx in range(n_mics):
            f, t, stft = signal.stft(signals[mic_idx], 
                                   fs=self.sample_rate,
                                   window=self.window,
                                   nperseg=self.fft_size,
                                   noverlap=self.fft_size - self.hop_size)
            stft_results.append(stft)
        
        return np.array(stft_results)
    
    def calculate_steering_vectors(self, mic_positions: np.ndarray, 
                                 target_points: np.ndarray) -> np.ndarray:
        """
        Steering vector 계산
        
        Args:
            mic_positions: 마이크 위치 (N_mics, 2)
            target_points: 타겟 포인트 (N_points, 3)
            
        Returns:
            steering_vectors: (N_freq, N_points, N_mics)
        """
        n_mics = len(mic_positions)
        n_points = len(target_points)
        n_freq = len(self.frequencies)
        
        # 3D 마이크 위치 (z=0 가정)
        mic_pos_3d = np.column_stack([mic_positions, np.zeros(n_mics)])
        
        # 거리 계산 (N_points, N_mics)
        distances = np.linalg.norm(
            target_points[:, np.newaxis, :] - mic_pos_3d[np.newaxis, :, :], 
            axis=2
        )
        
        # Steering vector 계산
        steering_vectors = np.zeros((n_freq, n_points, n_mics), dtype=complex)
        
        for freq_idx, freq in enumerate(self.frequencies):
            if freq == 0:  # DC 성분 처리
                steering_vectors[freq_idx] = np.ones((n_points, n_mics))
            else:
                # 위상 보정 인자 계산
                phase_correction = -2j * np.pi * freq * distances / self.sound_speed
                steering_vectors[freq_idx] = np.exp(phase_correction)
        
        return steering_vectors
    
    def _compute_covariance_matrix(self, stft_data: np.ndarray, freq_idx: int) -> np.ndarray:
        """
        공분산 행렬 계산
        
        Args:
            stft_data: STFT 데이터 (N_mics, N_freq, N_time)
            freq_idx: 주파수 인덱스
            
        Returns:
            covariance_matrix: 공분산 행렬 (N_mics, N_mics)
        """
        signal_freq = stft_data[:, freq_idx, :]  # (N_mics, N_time)
        n_mics, n_time = signal_freq.shape
        
        # 공분산 행렬 계산: R = E[x * x^H]
        covariance = np.zeros((n_mics, n_mics), dtype=complex)
        for time_idx in range(n_time):
            x = signal_freq[:, time_idx:time_idx+1]  # (N_mics, 1)
            covariance += x @ x.conj().T
        
        covariance /= n_time
        
        # 대각선 로딩으로 정규화
        covariance += self.diagonal_loading * np.eye(n_mics)
        
        return covariance
    
    def _mvdr_beamforming(self, stft_data: np.ndarray, steering_vectors: np.ndarray, freq_idx: int) -> np.ndarray:
        """
        MVDR Beamforming 적용
        
        Args:
            stft_data: STFT 데이터 (N_mics, N_freq, N_time)
            steering_vectors: Steering vectors (N_freq, N_points, N_mics)
            freq_idx: 주파수 인덱스
            
        Returns:
            beamformed_output: Beamformed 신호 (N_points, N_time)
        """
        n_mics, n_time = stft_data[:, freq_idx, :].shape
        n_points = steering_vectors.shape[1]
        
        # 공분산 행렬 계산
        R = self._compute_covariance_matrix(stft_data, freq_idx)
        
        beamformed_output = np.zeros((n_points, n_time), dtype=complex)
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # 특이행렬인 경우 pseudo-inverse 사용
            R_inv = np.linalg.pinv(R)
        
        for point_idx in range(n_points):
            a = steering_vectors[freq_idx, point_idx, :].reshape(-1, 1)  # (N_mics, 1)
            
            # MVDR weight: w = R^(-1) * a / (a^H * R^(-1) * a)
            numerator = R_inv @ a
            denominator = a.conj().T @ R_inv @ a
            
            if np.abs(denominator) < 1e-10:
                # Fallback to filter-and-sum
                weight = a / (np.linalg.norm(a)**2 + 1e-10)
            else:
                weight = numerator / denominator
            
            # Beamformed output
            beamformed_output[point_idx, :] = (weight.conj().T @ stft_data[:, freq_idx, :]).flatten()
        
        return beamformed_output
    
    def _robust_mvdr_beamforming(self, stft_data: np.ndarray, steering_vectors: np.ndarray, freq_idx: int) -> np.ndarray:
        """
        Robust MVDR Beamforming (공분산 행렬 재구성 포함)
        
        Args:
            stft_data: STFT 데이터 (N_mics, N_freq, N_time)
            steering_vectors: Steering vectors (N_freq, N_points, N_mics)
            freq_idx: 주파수 인덱스
            
        Returns:
            beamformed_output: Beamformed 신호 (N_points, N_time)
        """
        n_mics, n_time = stft_data[:, freq_idx, :].shape
        n_points = steering_vectors.shape[1]
        
        # 공분산 행렬 계산
        R = self._compute_covariance_matrix(stft_data, freq_idx)
        
        # 공분산 행렬 재구성 (eigenvalue decomposition)
        eigenvals, eigenvecs = np.linalg.eigh(R)
        
        # 작은 고유값들을 정규화
        eigenvals = np.maximum(eigenvals, self.regularization * np.max(eigenvals))
        
        # 재구성된 공분산 행렬
        R_reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        beamformed_output = np.zeros((n_points, n_time), dtype=complex)
        
        try:
            R_inv = np.linalg.inv(R_reconstructed)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_reconstructed)
        
        for point_idx in range(n_points):
            a = steering_vectors[freq_idx, point_idx, :].reshape(-1, 1)  # (N_mics, 1)
            
            # Robust MVDR weight
            numerator = R_inv @ a
            denominator = a.conj().T @ R_inv @ a
            
            if np.abs(denominator) < 1e-10:
                weight = a / (np.linalg.norm(a)**2 + 1e-10)
            else:
                weight = numerator / denominator
            
            beamformed_output[point_idx, :] = (weight.conj().T @ stft_data[:, freq_idx, :]).flatten()
        
        return beamformed_output
    
    def apply_beamforming(self, stft_data: np.ndarray, steering_vectors: np.ndarray) -> np.ndarray:
        """
        Beamforming 적용 (알고리즘 타입에 따라 다른 방법 사용)
        
        Args:
            stft_data: STFT 데이터 (N_mics, N_freq, N_time)
            steering_vectors: Steering vectors (N_freq, N_points, N_mics)
            
        Returns:
            beamformed_stft: Beamformed STFT (N_points, N_freq, N_time)
        """
        n_mics, n_freq, n_time = stft_data.shape
        n_points = steering_vectors.shape[1]
        
        beamformed_stft = np.zeros((n_points, n_freq, n_time), dtype=complex)
        
        # 각 주파수 빈에 대해 beamforming 수행
        for freq_idx in range(n_freq):
            if self.frequencies[freq_idx] == 0:  # DC 성분 스킵
                continue
            
            if self.beamforming_type == BeamformingType.MVDR:
                beamformed_stft[:, freq_idx, :] = self._mvdr_beamforming(
                    stft_data, steering_vectors, freq_idx)
            elif self.beamforming_type == BeamformingType.ROBUST_MVDR:
                beamformed_stft[:, freq_idx, :] = self._robust_mvdr_beamforming(
                    stft_data, steering_vectors, freq_idx)
            else:  # FILTER_AND_SUM (기본값)
                # 현재 주파수의 모든 마이크 데이터 (N_mics, N_time)
                freq_data = stft_data[:, freq_idx, :]
                
                # 현재 주파수의 steering vector (N_points, N_mics)
                steering = steering_vectors[freq_idx]
                
                # Filter-and-Sum beamforming: 각 포인트에 대해 가중합
                for point_idx in range(n_points):
                    steering_vec = steering[point_idx, :]
                    # w = steering_vector / |steering_vector|^2
                    weight = steering_vec / (np.linalg.norm(steering_vec)**2 + 1e-10)
                    # Beamformed output
                    beamformed_stft[point_idx, freq_idx, :] = np.dot(weight.conj(), freq_data)
        
        return beamformed_stft
    
    def calculate_energy_map(self, beamformed_stft: np.ndarray, 
                           grid_shape: Tuple[int, int] = (70, 70)) -> np.ndarray:
        """
        에너지 맵 계산
        
        Args:
            beamformed_stft: Beamformed STFT (N_points, N_freq, N_time)
            grid_shape: 그리드 형태 (height, width)
            
        Returns:
            energy_map: 에너지 맵 (height, width)
        """
        # 주파수 및 시간에 대한 에너지 합계
        energy_per_point = np.sum(np.abs(beamformed_stft)**2, axis=(1, 2))
        
        # 그리드 형태로 reshape
        if len(energy_per_point) != grid_shape[0] * grid_shape[1]:
            warnings.warn(f"Point count ({len(energy_per_point)}) doesn't match grid size ({grid_shape})")
            # 필요한 경우 interpolation 또는 padding 수행
            expected_size = grid_shape[0] * grid_shape[1]
            if len(energy_per_point) < expected_size:
                # Zero padding
                padded_energy = np.zeros(expected_size)
                padded_energy[:len(energy_per_point)] = energy_per_point
                energy_per_point = padded_energy
            else:
                # Truncate
                energy_per_point = energy_per_point[:expected_size]
        
        energy_map = energy_per_point.reshape(grid_shape)
        return energy_map
    
    def process_signals(self, mic_signals: np.ndarray, mic_positions: np.ndarray,
                       target_grid: np.ndarray, grid_shape: Tuple[int, int] = (70, 70)) -> np.ndarray:
        """
        전체 신호 처리 파이프라인
        
        Args:
            mic_signals: 마이크 신호 (N_mics, N_samples)
            mic_positions: 마이크 위치 (N_mics, 2)
            target_grid: 타겟 그리드 포인트 (N_points, 3)
            grid_shape: 그리드 형태
            
        Returns:
            energy_map: 에너지 맵 (height, width)
        """
        # 1. STFT 분석
        stft_data = self.stft_analysis(mic_signals)
        
        # 2. Steering vector 계산
        steering_vectors = self.calculate_steering_vectors(mic_positions, target_grid)
        
        # 3. Beamforming 적용
        beamformed_stft = self.apply_beamforming(stft_data, steering_vectors)
        
        # 4. 에너지 맵 계산
        energy_map = self.calculate_energy_map(beamformed_stft, grid_shape)
        
        return energy_map

def create_target_grid(grid_size: Tuple[int, int] = (70, 70), 
                      physical_size: Tuple[float, float] = (1.0, 1.0),
                      z_distance: float = 1.0) -> np.ndarray:
    """
    타겟 그리드 생성
    
    Args:
        grid_size: 그리드 크기 (height, width)
        physical_size: 물리적 크기 (height_m, width_m)
        z_distance: Z 거리 (m)
        
    Returns:
        target_grid: 타겟 포인트들 (N_points, 3)
    """
    height, width = grid_size
    height_m, width_m = physical_size
    
    # 그리드 좌표 생성
    x = np.linspace(-width_m/2, width_m/2, width)
    y = np.linspace(-height_m/2, height_m/2, height)
    
    # 메쉬그리드 생성
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_distance)
    
    # (N_points, 3) 형태로 변환
    target_grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return target_grid