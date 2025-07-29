#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beam Focusing Tutorial - 교육용 단계별 설명
===========================================

이 파일은 오디오 빔 포커싱의 전체 과정을 단계별로 설명합니다.
각 단계마다 상세한 주석과 시각화를 포함하여 이해하기 쉽게 구성되었습니다.

주요 단계:
1. 오디오 데이터 생성
2. 마이크 배열 및 타겟 포인트 설정
3. Time Delay 계산
4. Phase Delay 적용
5. FFT 기반 합성곱
6. 에너지 맵 계산
7. 결과 시각화

작성자: AI Assistant
목적: 교육용 자료
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import time

class BeamFocusingTutorial:
    """
    빔 포커싱 교육용 클래스
    각 단계를 명확히 분리하여 이해하기 쉽게 구성
    """
    
    def __init__(self):
        # 기본 파라미터 설정
        self.sample_rate = 44100  # 샘플링 주파수 (Hz)
        self.sound_speed = 343.0  # 음속 (m/s)
        
        # 마이크 배열 설정
        self.n_mics = 8           # 마이크 개수
        self.array_radius = 0.1   # 배열 반지름 (m)
        
        # 타겟 영역 설정
        self.target_distance = 0.3  # 타겟 평면까지의 거리 (m)
        self.target_size = 0.2      # 타겟 영역 크기 (m)
        self.grid_resolution = 20   # 그리드 해상도
        
        print("🎯 Beam Focusing Tutorial 초기화 완료")
        print(f"   - 마이크 개수: {self.n_mics}개")
        print(f"   - 배열 반지름: {self.array_radius}m")
        print(f"   - 타겟 거리: {self.target_distance}m")
        print(f"   - 샘플링 주파수: {self.sample_rate}Hz")
        print()
    
    def step1_generate_audio_data(self, duration=2.0, frequency=1000.0, source_trajectory=None):
        """
        단계 1: 오디오 데이터 생성 (시간에 따른 소스 이동 포함)
        
        Args:
            duration: 신호 길이 (초)
            frequency: 신호 주파수 (Hz)
            source_trajectory: 소스 궤적 함수 또는 None (정적 소스)
        
        Returns:
            audio_data: (n_samples, n_mics) 형태의 오디오 데이터
        """
        print("📡 단계 1: 오디오 데이터 생성")
        print("=" * 40)
        
        # 시간 축 생성
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        print(f"   - 신호 길이: {duration}초 ({n_samples} 샘플)")
        print(f"   - 신호 주파수: {frequency}Hz")
        # 소스 궤적 설정
        if source_trajectory is None:
            # 기본: 원형 궤적으로 이동하는 소스
            def default_trajectory(t):
                radius = 0.06
                angular_freq = 2 * np.pi / duration  # 한 바퀴 회전
                x = radius * np.cos(angular_freq * t)
                y = radius * np.sin(angular_freq * t)
                z = self.target_distance
                return np.array([x, y, z])
            source_trajectory = default_trajectory
        
        print(f"   - 소스 궤적: 시간에 따라 이동하는 소스")
        
        # 마이크 위치 설정 (원형 배열)
        angles = np.linspace(0, 2*np.pi, self.n_mics, endpoint=False)
        mic_positions = np.column_stack([
            self.array_radius * np.cos(angles),
            self.array_radius * np.sin(angles),
            np.zeros(self.n_mics)  # 모든 마이크가 z=0 평면에 위치
        ])
        
        print(f"   - 마이크 배열: 원형, 반지름 {self.array_radius}m")
        
        # 시간에 따른 소스 위치 계산
        source_positions = np.array([source_trajectory(time) for time in t])
        
        # 기본 신호 생성 (사인파)
        base_signal = np.sin(2 * np.pi * frequency * t)
        
        # 각 마이크에 도달하는 신호 생성 (시간에 따른 지연 및 감쇠 적용)
        audio_data = np.zeros((n_samples, self.n_mics))
        
        print(f"   - 시간에 따른 신호 생성 중...")
        
        for sample_idx in range(n_samples):
            current_source_pos = source_positions[sample_idx]
            
            # 각 마이크까지의 거리 계산
            distances = np.sqrt(np.sum((mic_positions - current_source_pos)**2, axis=1))
            
            # 전파 지연 시간 계산
            propagation_delays = distances / self.sound_speed
            
            for mic_idx in range(self.n_mics):
                # 거리에 따른 감쇠 (1/r 법칙)
                amplitude = 1.0 / distances[mic_idx]
                
                # 전파 지연 적용
                delay_samples = int(propagation_delays[mic_idx] * self.sample_rate)
                source_sample_idx = sample_idx - delay_samples
                
                if source_sample_idx >= 0:
                    audio_data[sample_idx, mic_idx] = amplitude * base_signal[source_sample_idx]
        
        # 노이즈 추가 (현실적인 시뮬레이션)
        noise_level = 0.01
        audio_data += noise_level * np.random.randn(n_samples, self.n_mics)
        
        print(f"   - 노이즈 레벨: {noise_level}")
        print(f"   - 생성된 데이터 크기: {audio_data.shape}")
        print("   ✅ 오디오 데이터 생성 완료\n")
        
        # 결과 저장 (다음 단계에서 사용)
        self.audio_data = audio_data
        self.mic_positions = mic_positions
        self.source_positions = source_positions  # 시간에 따른 소스 위치들
        self.source_trajectory = source_trajectory
        self.time_axis = t
        
        print(f"   - 소스 이동 범위: X[{np.min(source_positions[:, 0]):.3f}, {np.max(source_positions[:, 0]):.3f}]m")
        print(f"                    Y[{np.min(source_positions[:, 1]):.3f}, {np.max(source_positions[:, 1]):.3f}]m")
        
        return audio_data
    
    def step2_setup_target_points(self):
        """
        단계 2: 타겟 포인트 설정
        
        Returns:
            target_points: (n_points, 3) 형태의 타겟 포인트 좌표
        """
        print("🎯 단계 2: 타겟 포인트 설정")
        print("=" * 40)
        
        # 타겟 평면의 그리드 생성
        x_grid = np.linspace(-self.target_size/2, self.target_size/2, self.grid_resolution)
        y_grid = np.linspace(-self.target_size/2, self.target_size/2, self.grid_resolution)
        
        # meshgrid를 사용하여 2D 그리드 생성
        Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')
        
        # 3D 좌표로 변환 (z = target_distance)
        target_points = np.column_stack([
            X.flatten(),
            Y.flatten(),
            np.full(X.size, self.target_distance)
        ])
        
        print(f"   - 타겟 평면 크기: {self.target_size}m × {self.target_size}m")
        print(f"   - 타겟 평면 거리: {self.target_distance}m")
        print(f"   - 그리드 해상도: {self.grid_resolution}×{self.grid_resolution}")
        print(f"   - 총 타겟 포인트: {len(target_points)}개")
        print("   ✅ 타겟 포인트 설정 완료\n")
        
        self.target_points = target_points
        return target_points
    
    def step3_calculate_time_delays(self):
        """
        단계 3: Time Delay 계산
        
        각 타겟 포인트에서 각 마이크까지의 전파 시간을 계산합니다.
        
        Returns:
            time_delays: (n_target_points, n_mics) 형태의 시간 지연
        """
        print("⏱️ 단계 3: Time Delay 계산")
        print("=" * 40)
        
        n_target_points = len(self.target_points)
        time_delays = np.zeros((n_target_points, self.n_mics))
        
        print(f"   - 계산할 조합: {n_target_points} 타겟 × {self.n_mics} 마이크")
        
        # 각 타겟 포인트에 대해
        for target_idx, target_point in enumerate(self.target_points):
            # 각 마이크까지의 거리 계산
            distances = np.sqrt(np.sum((self.mic_positions - target_point)**2, axis=1))
            
            # 시간 지연 계산 (거리 / 음속)
            delays = distances / self.sound_speed
            
            # 상대적 지연 계산 (최소 지연을 기준으로)
            relative_delays = delays - np.min(delays)
            
            time_delays[target_idx, :] = relative_delays
        
        # 통계 정보 출력
        max_delay = np.max(time_delays)
        mean_delay = np.mean(time_delays)
        
        print(f"   - 최대 상대 지연: {max_delay*1000:.3f}ms")
        print(f"   - 평균 상대 지연: {mean_delay*1000:.3f}ms")
        print(f"   - 최대 지연 샘플: {int(max_delay * self.sample_rate)}")
        print("   ✅ Time Delay 계산 완료\n")
        
        self.time_delays = time_delays
        return time_delays
    
    def step4_apply_phase_delays(self, target_idx=None):
        """
        단계 4: Phase Delay 적용
        
        특정 타겟 포인트에 대해 각 마이크 신호에 위상 지연을 적용합니다.
        
        Args:
            target_idx: 타겟 포인트 인덱스 (None이면 중앙 포인트 사용)
        
        Returns:
            delayed_signals: (n_samples, n_mics) 형태의 지연된 신호
        """
        print("🔄 단계 4: Phase Delay 적용")
        print("=" * 40)
        
        if target_idx is None:
            # 중앙 포인트 선택
            target_idx = len(self.target_points) // 2
        
        target_point = self.target_points[target_idx]
        delays = self.time_delays[target_idx, :]
        
        print(f"   - 선택된 타겟 포인트: {target_point}")
        print(f"   - 타겟 인덱스: {target_idx}")
        
        n_samples, n_mics = self.audio_data.shape
        delayed_signals = np.zeros_like(self.audio_data)
        
        # 각 마이크 신호에 지연 적용
        for mic_idx in range(n_mics):
            delay_time = delays[mic_idx]
            delay_samples = delay_time * self.sample_rate
            
            # 정수 부분과 소수 부분 분리
            int_delay = int(delay_samples)
            frac_delay = delay_samples - int_delay
            
            print(f"   - 마이크 {mic_idx}: {delay_time*1000:.3f}ms ({delay_samples:.2f} 샘플)")
            
            # 정수 지연 적용
            if int_delay < n_samples:
                shifted_signal = np.zeros(n_samples)
                shifted_signal[int_delay:] = self.audio_data[:n_samples-int_delay, mic_idx]
                
                # 소수 지연 적용 (선형 보간)
                if frac_delay > 0 and int_delay + 1 < n_samples:
                    shifted_signal[int_delay+1:] = (
                        (1 - frac_delay) * self.audio_data[:n_samples-int_delay-1, mic_idx] +
                        frac_delay * self.audio_data[1:n_samples-int_delay, mic_idx]
                    )
                
                delayed_signals[:, mic_idx] = shifted_signal
        
        print("   ✅ Phase Delay 적용 완료\n")
        
        self.delayed_signals = delayed_signals
        self.current_target_idx = target_idx
        return delayed_signals
    
    def step5_fft_convolution(self):
        """
        단계 5: FFT 기반 합성곱
        
        지연된 신호들을 FFT 도메인에서 처리하고 합성합니다.
        
        Returns:
            focused_signal: 포커싱된 신호
            fft_data: FFT 분석 데이터
        """
        print("🔄 단계 5: FFT 기반 합성곱")
        print("=" * 40)
        
        n_samples, n_mics = self.delayed_signals.shape
        
        # FFT 크기 결정 (2의 거듭제곱으로 패딩)
        fft_size = 2 ** int(np.ceil(np.log2(n_samples)))
        print(f"   - 원본 신호 길이: {n_samples} 샘플")
        print(f"   - FFT 크기: {fft_size} 샘플")
        
        # 주파수 축 생성
        freqs = fftfreq(fft_size, 1/self.sample_rate)
        
        # 각 마이크 신호의 FFT 계산
        fft_signals = np.zeros((fft_size, n_mics), dtype=complex)
        
        for mic_idx in range(n_mics):
            # 제로 패딩 후 FFT
            padded_signal = np.zeros(fft_size)
            padded_signal[:n_samples] = self.delayed_signals[:, mic_idx]
            fft_signals[:, mic_idx] = fft(padded_signal)
        
        print(f"   - FFT 계산 완료: {n_mics}개 마이크")
        
        # 신호 합성 (Delay-and-Sum)
        combined_fft = np.sum(fft_signals, axis=1)
        
        # 역 FFT로 시간 도메인 복원
        focused_signal = np.real(ifft(combined_fft))[:n_samples]
        
        print(f"   - 합성된 신호 길이: {len(focused_signal)} 샘플")
        
        # FFT 분석 데이터 저장
        fft_data = {
            'frequencies': freqs[:fft_size//2],
            'magnitude': np.abs(combined_fft[:fft_size//2]),
            'phase': np.angle(combined_fft[:fft_size//2]),
            'individual_ffts': fft_signals[:fft_size//2, :]
        }
        
        print("   ✅ FFT 기반 합성곱 완료\n")
        
        self.focused_signal = focused_signal
        self.fft_data = fft_data
        return focused_signal, fft_data
    
    def step6_compute_energy_map_time_series(self, time_window=0.2, overlap=0.5):
        """
        단계 6: 시간에 따른 에너지 맵 계산
        
        시간 윈도우를 사용하여 소스의 이동을 추적합니다.
        
        Args:
            time_window: 분석 윈도우 크기 (초)
            overlap: 윈도우 겹침 비율
        
        Returns:
            energy_maps: (n_time_steps, grid_resolution, grid_resolution) 형태의 에너지 맵들
            time_stamps: 각 에너지 맵의 시간 스탬프
        """
        print("⚡ 단계 6: 시간에 따른 에너지 맵 계산")
        print("=" * 40)
        
        # 시간 윈도우 설정
        window_samples = int(time_window * self.sample_rate)
        hop_samples = int(window_samples * (1 - overlap))
        n_time_steps = (len(self.audio_data) - window_samples) // hop_samples + 1
        
        print(f"   - 시간 윈도우: {time_window}초 ({window_samples} 샘플)")
        print(f"   - 겹침 비율: {overlap*100:.0f}%")
        print(f"   - 시간 스텝: {n_time_steps}개")
        
        n_target_points = len(self.target_points)
        energy_maps = np.zeros((n_time_steps, self.grid_resolution, self.grid_resolution))
        time_stamps = np.zeros(n_time_steps)
        
        print(f"   - 계산할 조합: {n_time_steps} 시간 × {n_target_points} 타겟")
        
        # 각 시간 윈도우에 대해
        for t_idx in range(n_time_steps):
            start_sample = t_idx * hop_samples
            end_sample = start_sample + window_samples
            time_stamps[t_idx] = start_sample / self.sample_rate
            
            # 현재 윈도우의 오디오 데이터
            window_data = self.audio_data[start_sample:end_sample, :]
            
            energy_values = np.zeros(n_target_points)
            
            # 각 타겟 포인트에 대해 에너지 계산
            for target_idx in range(n_target_points):
                # 해당 타겟에 대한 지연 적용
                delays = self.time_delays[target_idx, :]
                
                # 지연된 신호들 합성
                focused_signal = np.zeros(window_samples)
                
                for mic_idx in range(self.n_mics):
                    delay_samples = int(delays[mic_idx] * self.sample_rate)
                    
                    if delay_samples < window_samples:
                        # 간단한 정수 지연 적용
                        shifted_signal = np.zeros(window_samples)
                        shifted_signal[delay_samples:] = window_data[:window_samples-delay_samples, mic_idx]
                        focused_signal += shifted_signal
                
                # RMS 에너지 계산
                energy_values[target_idx] = np.sqrt(np.mean(focused_signal**2))
            
            # 2D 그리드로 변환 및 정규화
            energy_map = energy_values.reshape(self.grid_resolution, self.grid_resolution)
            if np.max(energy_map) > 0:
                energy_map = energy_map / np.max(energy_map)
            
            energy_maps[t_idx] = energy_map
            
            if (t_idx + 1) % 5 == 0:
                print(f"   - 진행률: {t_idx + 1}/{n_time_steps} ({100*(t_idx+1)/n_time_steps:.1f}%)")
        
        print(f"   - 에너지 맵 시리즈 크기: {energy_maps.shape}")
        print("   ✅ 시간에 따른 에너지 맵 계산 완료\n")
        
        self.energy_maps = energy_maps
        self.time_stamps = time_stamps
        return energy_maps, time_stamps
    
    def step7_visualize_results(self):
        """
        단계 7: 결과 시각화 (시간에 따른 변화 포함)
        
        모든 단계의 결과를 종합적으로 시각화하며, 특히 시간에 따른 소스 추적을 강조합니다.
        """
        print("📊 단계 7: 결과 시각화")
        print("=" * 40)
        
        # 큰 figure 생성
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle('Beam Focusing Tutorial - 시간에 따른 소스 추적', fontsize=18, fontweight='bold')
        
        # 1. 마이크 배열과 소스 궤적
        ax1 = plt.subplot(4, 5, 1)
        ax1.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                   c='blue', s=100, marker='o', label='마이크')
        
        # 소스 궤적 표시
        ax1.plot(self.source_positions[:, 0], self.source_positions[:, 1], 
                'r-', linewidth=2, alpha=0.7, label='소스 궤적')
        ax1.scatter(self.source_positions[0, 0], self.source_positions[0, 1], 
                   c='green', s=150, marker='o', label='시작점')
        ax1.scatter(self.source_positions[-1, 0], self.source_positions[-1, 1], 
                   c='red', s=150, marker='s', label='끝점')
        
        # 타겟 영역 표시
        target_x = self.target_points[:, 0].reshape(self.grid_resolution, self.grid_resolution)
        target_y = self.target_points[:, 1].reshape(self.grid_resolution, self.grid_resolution)
        ax1.contour(target_x, target_y, np.ones_like(target_x), levels=[0.5], colors='gray', alpha=0.5)
        
        ax1.set_xlim(-0.15, 0.15)
        ax1.set_ylim(-0.15, 0.15)
        ax1.set_aspect('equal')
        ax1.set_title('1. 시스템 구성 및 소스 궤적')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. 원본 오디오 신호 (시간에 따른 변화)
        ax2 = plt.subplot(4, 5, 2)
        time_ms = self.time_axis * 1000
        for i in range(min(3, self.n_mics)):
            ax2.plot(time_ms[:2000], self.audio_data[:2000, i], alpha=0.7, label=f'마이크 {i+1}')
        ax2.set_xlabel('시간 (ms)')
        ax2.set_ylabel('진폭')
        ax2.set_title('2. 원본 오디오 신호')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Time Delay 분포 (중앙 타겟)
        ax3 = plt.subplot(4, 5, 3)
        delay_map = self.time_delays[self.current_target_idx, :]
        bars = ax3.bar(range(self.n_mics), delay_map * 1000)
        ax3.set_xlabel('마이크 번호')
        ax3.set_ylabel('지연 시간 (ms)')
        ax3.set_title('3. Time Delay 분포')
        ax3.grid(True, alpha=0.3)
        
        # 4. 지연된 신호
        ax4 = plt.subplot(4, 5, 4)
        for i in range(min(3, self.n_mics)):
            ax4.plot(time_ms[:1000], self.delayed_signals[:1000, i], alpha=0.7, label=f'마이크 {i+1}')
        ax4.set_xlabel('시간 (ms)')
        ax4.set_ylabel('진폭')
        ax4.set_title('4. 지연된 신호')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. FFT 스펙트럼
        ax5 = plt.subplot(4, 5, 5)
        freqs_khz = self.fft_data['frequencies'] / 1000
        ax5.semilogy(freqs_khz, self.fft_data['magnitude'])
        ax5.set_xlabel('주파수 (kHz)')
        ax5.set_ylabel('크기')
        ax5.set_title('5. FFT 스펙트럼')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 5)
        
        # 6. 포커싱된 신호
        ax6 = plt.subplot(4, 5, 6)
        ax6.plot(time_ms[:1000], self.focused_signal[:1000], 'g-', linewidth=2)
        ax6.set_xlabel('시간 (ms)')
        ax6.set_ylabel('진폭')
        ax6.set_title('6. 포커싱된 신호')
        ax6.grid(True, alpha=0.3)
        
        # 7-10. 시간에 따른 에너지 맵 (4개 시점)
        extent = [-self.target_size/2, self.target_size/2, -self.target_size/2, self.target_size/2]
        time_indices = [0, len(self.energy_maps)//3, 2*len(self.energy_maps)//3, -1]
        titles = ['초기', '중간1', '중간2', '최종']
        
        for i, (t_idx, title) in enumerate(zip(time_indices, titles)):
            ax = plt.subplot(4, 5, 7 + i)
            im = ax.imshow(self.energy_maps[t_idx], extent=extent, origin='lower', cmap='hot')
            
            # 해당 시간의 실제 소스 위치
            time_sample = int(self.time_stamps[t_idx] * self.sample_rate)
            if time_sample < len(self.source_positions):
                true_pos = self.source_positions[time_sample]
                ax.scatter(true_pos[0], true_pos[1], c='white', s=150, 
                          marker='*', edgecolors='black', linewidths=2, label='실제 위치')
            
            # 감지된 피크 위치
            peak_idx = np.unravel_index(np.argmax(self.energy_maps[t_idx]), self.energy_maps[t_idx].shape)
            detected_x = (peak_idx[1] / self.grid_resolution - 0.5) * self.target_size
            detected_y = (peak_idx[0] / self.grid_resolution - 0.5) * self.target_size
            ax.scatter(detected_x, detected_y, c='cyan', s=100, 
                      marker='o', edgecolors='blue', linewidths=2, label='감지 위치')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{7+i}. {title} (t={self.time_stamps[t_idx]:.2f}s)')
            if i == 0:
                ax.legend(fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # 11. 소스 추적 결과
        ax11 = plt.subplot(4, 5, 11)
        
        # 실제 궤적
        ax11.plot(self.source_positions[:, 0], self.source_positions[:, 1], 
                 'r-', linewidth=3, alpha=0.8, label='실제 궤적')
        
        # 감지된 궤적
        detected_positions = []
        for t_idx in range(len(self.energy_maps)):
            peak_idx = np.unravel_index(np.argmax(self.energy_maps[t_idx]), self.energy_maps[t_idx].shape)
            detected_x = (peak_idx[1] / self.grid_resolution - 0.5) * self.target_size
            detected_y = (peak_idx[0] / self.grid_resolution - 0.5) * self.target_size
            detected_positions.append([detected_x, detected_y])
        
        detected_positions = np.array(detected_positions)
        ax11.plot(detected_positions[:, 0], detected_positions[:, 1], 
                 'b--', linewidth=2, alpha=0.8, label='감지된 궤적')
        
        # 마이크 위치
        ax11.scatter(self.mic_positions[:, 0], self.mic_positions[:, 1], 
                    c='gray', s=50, marker='o', alpha=0.6, label='마이크')
        
        ax11.set_xlim(-0.15, 0.15)
        ax11.set_ylim(-0.15, 0.15)
        ax11.set_aspect('equal')
        ax11.set_xlabel('X (m)')
        ax11.set_ylabel('Y (m)')
        ax11.set_title('11. 소스 추적 결과')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)
        
        # 12. 추적 오차 분석
        ax12 = plt.subplot(4, 5, 12)
        
        # 시간에 따른 추적 오차 계산
        tracking_errors = []
        for t_idx in range(len(self.energy_maps)):
            time_sample = int(self.time_stamps[t_idx] * self.sample_rate)
            if time_sample < len(self.source_positions):
                true_pos = self.source_positions[time_sample]
                detected_pos = detected_positions[t_idx]
                error = np.sqrt((true_pos[0] - detected_pos[0])**2 + (true_pos[1] - detected_pos[1])**2)
                tracking_errors.append(error * 1000)  # mm 단위
        
        ax12.plot(self.time_stamps[:len(tracking_errors)], tracking_errors, 'g-', linewidth=2)
        ax12.set_xlabel('시간 (s)')
        ax12.set_ylabel('추적 오차 (mm)')
        ax12.set_title('12. 시간에 따른 추적 오차')
        ax12.grid(True, alpha=0.3)
        
        # 평균 오차 표시
        mean_error = np.mean(tracking_errors)
        ax12.axhline(y=mean_error, color='r', linestyle='--', alpha=0.7, 
                    label=f'평균: {mean_error:.1f}mm')
        ax12.legend(fontsize=8)
        
        # 13. 에너지 시간 변화
        ax13 = plt.subplot(4, 5, 13)
        
        # 각 시간에서의 최대 에너지
        max_energies = [np.max(energy_map) for energy_map in self.energy_maps]
        ax13.plot(self.time_stamps, max_energies, 'b-', linewidth=2, label='최대 에너지')
        
        # 각 시간에서의 평균 에너지
        mean_energies = [np.mean(energy_map) for energy_map in self.energy_maps]
        ax13.plot(self.time_stamps, mean_energies, 'r--', linewidth=2, label='평균 에너지')
        
        ax13.set_xlabel('시간 (s)')
        ax13.set_ylabel('에너지')
        ax13.set_title('13. 시간에 따른 에너지 변화')
        ax13.legend(fontsize=8)
        ax13.grid(True, alpha=0.3)
        
        # 14. 소스 속도 분석
        ax14 = plt.subplot(4, 5, 14)
        
        # 실제 속도 계산
        dt = np.diff(self.time_axis[:len(self.source_positions)])
        dx = np.diff(self.source_positions[:, 0])
        dy = np.diff(self.source_positions[:, 1])
        velocities = np.sqrt(dx**2 + dy**2) / dt[:len(dx)]
        
        time_vel = self.time_axis[:len(velocities)]
        ax14.plot(time_vel, velocities, 'purple', linewidth=2)
        ax14.set_xlabel('시간 (s)')
        ax14.set_ylabel('속도 (m/s)')
        ax14.set_title('14. 소스 이동 속도')
        ax14.grid(True, alpha=0.3)
        
        # 15. 3D 에너지 맵 (최종)
        ax15 = plt.subplot(4, 5, 15, projection='3d')
        X, Y = np.meshgrid(np.linspace(-self.target_size/2, self.target_size/2, self.grid_resolution),
                          np.linspace(-self.target_size/2, self.target_size/2, self.grid_resolution))
        final_energy_map = self.energy_maps[-1]
        surf = ax15.plot_surface(X, Y, final_energy_map, cmap='hot', alpha=0.8)
        ax15.set_xlabel('X (m)')
        ax15.set_ylabel('Y (m)')
        ax15.set_zlabel('에너지')
        ax15.set_title('15. 3D 에너지 맵 (최종)')
        
        # 16-20. 성능 지표 및 통계
        ax16 = plt.subplot(4, 5, 16)
        ax16.axis('off')
        
        # 전체 추적 성능 계산
        mean_tracking_error = np.mean(tracking_errors)
        max_tracking_error = np.max(tracking_errors)
        min_tracking_error = np.min(tracking_errors)
        
        info_text = f"""
        📊 시간 추적 성능 지표
        
        추적 시간: {self.time_stamps[-1]:.2f}초
        시간 스텝: {len(self.energy_maps)}개
        
        평균 추적 오차: {mean_tracking_error:.1f}mm
        최대 추적 오차: {max_tracking_error:.1f}mm
        최소 추적 오차: {min_tracking_error:.1f}mm
        
        평균 소스 속도: {np.mean(velocities):.3f}m/s
        최대 소스 속도: {np.max(velocities):.3f}m/s
        
        에너지 맵 해상도: {self.grid_resolution}×{self.grid_resolution}
        """
        
        ax16.text(0.05, 0.95, info_text, transform=ax16.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 나머지 서브플롯들은 빈 공간으로 남겨둠
        for i in range(17, 21):
            ax = plt.subplot(4, 5, i)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('beam_focusing_tutorial_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ 시각화 완료")
        print(f"   - 결과 저장: beam_focusing_tutorial_results.png")
        print(f"   - 평균 추적 오차: {mean_tracking_error:.1f}mm")
        print(f"   - 추적 시간: {self.time_stamps[-1]:.2f}초")
        print()
    
    def run_complete_tutorial(self):
        """
        전체 튜토리얼 실행
        
        모든 단계를 순차적으로 실행하고 결과를 시각화합니다.
        """
        print("🚀 Beam Focusing 완전 튜토리얼 시작")
        print("=" * 50)
        print()
        
        start_time = time.time()
        
        # 단계별 실행
        self.step1_generate_audio_data()
        self.step2_setup_target_points()
        self.step3_calculate_time_delays()
        self.step4_apply_phase_delays()
        self.step5_fft_convolution()
        self.step6_compute_energy_map_time_series()
        self.step7_visualize_results()
        
        total_time = time.time() - start_time
        
        print("🎉 튜토리얼 완료!")
        print("=" * 50)
        print(f"총 실행 시간: {total_time:.2f}초")
        print()
        print("📚 학습 포인트:")
        print("   1. 오디오 신호는 거리에 따라 지연되고 감쇠됩니다")
        print("   2. Time delay 계산이 빔 포커싱의 핵심입니다")
        print("   3. FFT를 사용하면 효율적인 신호 처리가 가능합니다")
        print("   4. Delay-and-Sum 알고리즘으로 공간적 포커싱을 구현합니다")
        print("   5. 시간 윈도우를 통해 이동하는 소스를 추적할 수 있습니다")
        print("   6. 에너지 맵 시리즈로 소스의 궤적을 시각화할 수 있습니다")
        print("   7. 추적 오차 분석을 통해 시스템 성능을 평가할 수 있습니다")
        print()
        print("🔬 추가 실험 아이디어:")
        print("   - 다른 궤적 패턴 (직선, 나선형, 랜덤 등)")
        print("   - 다양한 이동 속도 테스트")
        print("   - 마이크 개수 및 배열 형태 변경")
        print("   - 시간 윈도우 크기 최적화")
        print("   - 다중 소스 동시 추적")
        print("   - 칼만 필터 등 고급 추적 알고리즘")
        print("   - 실시간 처리 성능 최적화")


def main():
    """
    메인 함수 - 튜토리얼 실행
    """
    print("🎓 Audio Beam Focusing Tutorial")
    print("교육용 단계별 설명 프로그램")
    print("=" * 50)
    print()
    
    # 튜토리얼 인스턴스 생성
    tutorial = BeamFocusingTutorial()
    
    # 완전 튜토리얼 실행
    tutorial.run_complete_tutorial()
    
    print("\n📖 이 코드를 통해 배울 수 있는 것:")
    print("   - 오디오 신호 처리의 기본 원리")
    print("   - 공간 음향학의 기초 개념")
    print("   - FFT와 디지털 신호 처리")
    print("   - 배열 신호 처리 (Array Signal Processing)")
    print("   - 빔포밍 알고리즘의 구현")
    print("   - 과학적 시각화 기법")


if __name__ == "__main__":
    main()