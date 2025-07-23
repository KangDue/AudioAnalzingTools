"""가상 음원 및 마이크 신호 시뮬레이션 모듈"""

import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

class SignalType(Enum):
    """신호 타입 열거형"""
    SINE_WAVE = "sine_wave"
    WHITE_NOISE = "white_noise"
    IMPULSE = "impulse"
    CHIRP = "chirp"

class AudioSource:
    """가상 음원 클래스"""
    
    def __init__(self, position: Tuple[float, float, float], signal_type: SignalType,
                 frequency: float = 1000.0, amplitude: float = 1.0, phase: float = 0.0):
        """
        음원 초기화
        
        Args:
            position: 음원 위치 (x, y, z)
            signal_type: 신호 타입
            frequency: 주파수 (Hz, sine_wave용)
            amplitude: 진폭
            phase: 위상 (라디안)
        """
        self.position = np.array(position)
        self.signal_type = signal_type
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
    
    def generate_signal(self, duration: float, sample_rate: int) -> np.ndarray:
        """
        신호 생성
        
        Args:
            duration: 신호 길이 (초)
            sample_rate: 샘플링 레이트 (Hz)
            
        Returns:
            signal: 생성된 신호
        """
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        if self.signal_type == SignalType.SINE_WAVE:
            signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
        elif self.signal_type == SignalType.WHITE_NOISE:
            signal = self.amplitude * np.random.randn(len(t))
        elif self.signal_type == SignalType.IMPULSE:
            signal = np.zeros(len(t))
            signal[0] = self.amplitude
        elif self.signal_type == SignalType.CHIRP:
            # Linear chirp from frequency/2 to frequency*2
            f0, f1 = self.frequency/2, self.frequency*2
            signal = self.amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
        else:
            raise ValueError(f"Unsupported signal type: {self.signal_type}")
            
        return signal

class SignalSimulator:
    """마이크 신호 시뮬레이션 클래스"""
    
    def __init__(self, sound_speed: float = 343.0):
        """
        시뮬레이터 초기화
        
        Args:
            sound_speed: 음속 (m/s)
        """
        self.sound_speed = sound_speed
        self.sources = []
    
    def add_source(self, source: AudioSource):
        """음원 추가"""
        self.sources.append(source)
    
    def simulate_mic_signals(self, mic_positions: np.ndarray, duration: float, 
                           sample_rate: int, add_noise: bool = True, 
                           noise_level: float = 0.01) -> np.ndarray:
        """
        각 마이크에서 수신되는 신호 시뮬레이션
        
        Args:
            mic_positions: 마이크 위치 (N_mics, 2) - 2D 좌표, z=0 가정
            duration: 신호 길이 (초)
            sample_rate: 샘플링 레이트 (Hz)
            add_noise: 노이즈 추가 여부
            noise_level: 노이즈 레벨
            
        Returns:
            mic_signals: 마이크 신호 (N_mics, N_samples)
        """
        n_samples = int(duration * sample_rate)
        n_mics = len(mic_positions)
        mic_signals = np.zeros((n_mics, n_samples))
        
        # 각 음원에 대해 신호 계산
        for source in self.sources:
            source_signal = source.generate_signal(duration, sample_rate)
            
            # 각 마이크에 대해 지연 및 감쇠 적용
            for mic_idx, mic_pos in enumerate(mic_positions):
                # 3D 마이크 위치 (z=0 가정)
                mic_pos_3d = np.array([mic_pos[0], mic_pos[1], 0.0])
                
                # 거리 계산
                distance = np.linalg.norm(source.position - mic_pos_3d)
                
                # 지연 시간 계산
                delay_time = distance / self.sound_speed
                delay_samples = int(delay_time * sample_rate)
                
                # 감쇠 계산 (1/r)
                attenuation = 1.0 / max(distance, 0.01)  # 최소 거리 제한
                
                # 지연된 신호 생성
                delayed_signal = np.zeros(n_samples)
                if delay_samples < n_samples:
                    delayed_signal[delay_samples:] = source_signal[:n_samples-delay_samples]
                
                # 감쇠 적용하여 마이크 신호에 추가
                mic_signals[mic_idx] += attenuation * delayed_signal
        
        # 노이즈 추가
        if add_noise:
            noise = noise_level * np.random.randn(n_mics, n_samples)
            mic_signals += noise
            
        return mic_signals
    
    def clear_sources(self):
        """모든 음원 제거"""
        self.sources.clear()