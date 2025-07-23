#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Processing Module
WAV 파일 로드, STFT, Spectrum 계산 담당
"""

import numpy as np
import soundfile as sf
from scipy.signal import stft
from pathlib import Path
import logging


class AudioProcessor:
    """오디오 처리 클래스"""
    
    def __init__(self, 
                 stft_nperseg=2048, 
                 stft_noverlap=1024, 
                 stft_window='hann'):
        """
        초기화
        
        Args:
            stft_nperseg: STFT 윈도우 크기
            stft_noverlap: STFT 오버랩 크기
            stft_window: STFT 윈도우 함수
        """
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.stft_window = stft_window
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_wav(self, file_path):
        """
        WAV 파일 로드
        
        Args:
            file_path: WAV 파일 경로
            
        Returns:
            dict: {
                'raw_data': np.ndarray,  # [channels, samples] 또는 [samples] for mono
                'metadata': dict
            }
        """
        try:
            file_path = Path(file_path)
            
            # soundfile로 WAV 로드
            data, sample_rate = sf.read(str(file_path), dtype='float32')
            
            # 채널 차원 정리 (mono: [samples] -> [1, samples], stereo: [samples, 2] -> [2, samples])
            if data.ndim == 1:
                # Mono
                raw_data = data.reshape(1, -1)
                num_channels = 1
            else:
                # Multi-channel (transpose to [channels, samples])
                raw_data = data.T
                num_channels = raw_data.shape[0]
                
            # 메타데이터 생성
            metadata = {
                'filename': file_path.name,
                'sampling_rate': sample_rate,
                'num_channels': num_channels,
                'num_samples': raw_data.shape[1],
                'length_sec': raw_data.shape[1] / sample_rate,
                'bit_depth': 32,  # float32로 로드했으므로
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }
            
            self.logger.info(f"WAV 로드 완료: {file_path.name} "
                           f"({num_channels}ch, {sample_rate}Hz, {metadata['length_sec']:.2f}s)")
            
            return {
                'raw_data': raw_data.astype(np.float32),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"WAV 로드 실패 {file_path}: {str(e)}")
            return None
            
    def compute_stft(self, raw_data):
        """
        STFT 계산 (magnitude만 반환)
        
        Args:
            raw_data: np.ndarray, shape [channels, samples]
            
        Returns:
            np.ndarray: STFT magnitude, shape [channels, frames, freq_bins]
        """
        try:
            channels, samples = raw_data.shape
            
            # 각 채널별로 STFT 계산
            stft_results = []
            
            for ch in range(channels):
                # scipy.signal.stft 사용
                frequencies, times, stft_complex = stft(
                    raw_data[ch],
                    nperseg=self.stft_nperseg,
                    noverlap=self.stft_noverlap,
                    window=self.stft_window,
                    return_onesided=True
                )
                
                # magnitude만 계산 (복소수 -> 절댓값)
                stft_magnitude = np.abs(stft_complex).astype(np.float32)
                stft_results.append(stft_magnitude)
                
            # [channels, freq_bins, frames] -> [channels, frames, freq_bins]로 변환
            stft_data = np.stack(stft_results, axis=0)  # [channels, freq_bins, frames]
            stft_data = np.transpose(stft_data, (0, 2, 1))  # [channels, frames, freq_bins]
            
            self.logger.info(f"STFT 계산 완료: shape {stft_data.shape}")
            
            return stft_data
            
        except Exception as e:
            self.logger.error(f"STFT 계산 실패: {str(e)}")
            return None
            
    def compute_spectrum(self, raw_data):
        """
        전체 Spectrum 계산 (FFT magnitude)
        
        Args:
            raw_data: np.ndarray, shape [channels, samples]
            
        Returns:
            np.ndarray: Spectrum magnitude, shape [channels, freq_bins]
        """
        try:
            channels, samples = raw_data.shape
            
            # FFT 길이 결정 (2의 거듭제곱으로 패딩)
            fft_length = 2 ** int(np.ceil(np.log2(samples)))
            
            spectrum_results = []
            
            for ch in range(channels):
                # FFT 계산 (zero-padding)
                fft_complex = np.fft.rfft(raw_data[ch], n=fft_length)
                
                # magnitude만 계산
                spectrum_magnitude = np.abs(fft_complex).astype(np.float32)
                spectrum_results.append(spectrum_magnitude)
                
            spectrum_data = np.stack(spectrum_results, axis=0)  # [channels, freq_bins]
            
            self.logger.info(f"Spectrum 계산 완료: shape {spectrum_data.shape}")
            
            return spectrum_data
            
        except Exception as e:
            self.logger.error(f"Spectrum 계산 실패: {str(e)}")
            return None
            
    def get_frequency_axis(self, sample_rate, fft_length=None, stft_mode=False):
        """
        주파수 축 생성
        
        Args:
            sample_rate: 샘플링 레이트
            fft_length: FFT 길이 (None이면 STFT 설정 사용)
            stft_mode: STFT용 주파수 축인지 여부
            
        Returns:
            np.ndarray: 주파수 축 [Hz]
        """
        if stft_mode:
            # STFT용 주파수 축
            freq_bins = self.stft_nperseg // 2 + 1
            return np.linspace(0, sample_rate / 2, freq_bins)
        else:
            # Spectrum용 주파수 축
            if fft_length is None:
                fft_length = self.stft_nperseg
            freq_bins = fft_length // 2 + 1
            return np.linspace(0, sample_rate / 2, freq_bins)
            
    def get_time_axis(self, num_samples, sample_rate, stft_mode=False):
        """
        시간 축 생성
        
        Args:
            num_samples: 총 샘플 수
            sample_rate: 샘플링 레이트
            stft_mode: STFT용 시간 축인지 여부
            
        Returns:
            np.ndarray: 시간 축 [초]
        """
        if stft_mode:
            # STFT용 시간 축
            hop_length = self.stft_nperseg - self.stft_noverlap
            num_frames = (num_samples - self.stft_noverlap) // hop_length
            return np.arange(num_frames) * hop_length / sample_rate
        else:
            # Raw data용 시간 축
            return np.arange(num_samples) / sample_rate
            
    def validate_audio_data(self, audio_data):
        """
        오디오 데이터 유효성 검사
        
        Args:
            audio_data: load_wav()의 반환값
            
        Returns:
            bool: 유효성 여부
        """
        if not isinstance(audio_data, dict):
            return False
            
        required_keys = ['raw_data', 'metadata']
        if not all(key in audio_data for key in required_keys):
            return False
            
        raw_data = audio_data['raw_data']
        if not isinstance(raw_data, np.ndarray) or raw_data.ndim != 2:
            return False
            
        metadata = audio_data['metadata']
        required_meta_keys = ['sampling_rate', 'num_channels', 'num_samples']
        if not all(key in metadata for key in required_meta_keys):
            return False
            
        return True
        
    def get_audio_info_summary(self, audio_data):
        """
        오디오 데이터 요약 정보 생성
        
        Args:
            audio_data: load_wav()의 반환값
            
        Returns:
            str: 요약 정보 문자열
        """
        if not self.validate_audio_data(audio_data):
            return "Invalid audio data"
            
        meta = audio_data['metadata']
        raw_data = audio_data['raw_data']
        
        summary = (
            f"File: {meta['filename']}\n"
            f"Channels: {meta['num_channels']}\n"
            f"Sample Rate: {meta['sampling_rate']} Hz\n"
            f"Duration: {meta['length_sec']:.2f} sec\n"
            f"Samples: {meta['num_samples']:,}\n"
            f"Data Range: [{raw_data.min():.3f}, {raw_data.max():.3f}]"
        )
        
        return summary