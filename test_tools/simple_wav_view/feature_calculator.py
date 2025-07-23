#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Calculator Module
사용자 정의 Feature 계산 및 히스토그램 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tkinter import filedialog, messagebox
import logging
from typing import List, Dict, Any


class FeatureCalculator:
    """Feature 계산 클래스"""
    
    def __init__(self):
        """
        초기화
        """
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Feature 계산 결과 저장
        self.feature_results = []
        
    def cal_feature(self, stft: np.ndarray) -> float:
        """
        사용자 정의 Feature 계산 함수
        
        Args:
            stft: np.ndarray, shape [channels, frames, freq_bins] - STFT magnitude 값
            
        Returns:
            float: 계산된 Feature 값
            
        Note:
            이 함수는 사용자가 필요에 따라 수정할 수 있습니다.
            현재는 예시로 평균 에너지를 계산합니다.
        """
        try:
            # 예시 1: 전체 평균 에너지
            mean_energy = np.mean(stft)
            
            # 예시 2: 최대 에너지
            # max_energy = np.max(stft)
            
            # 예시 3: 에너지의 표준편차
            # energy_std = np.std(stft)
            
            # 예시 4: 특정 주파수 대역의 에너지 (예: 1-5kHz)
            # freq_bins = stft.shape[2]
            # low_freq_idx = int(freq_bins * 0.1)  # 대략 1kHz (assuming 10kHz max)
            # high_freq_idx = int(freq_bins * 0.5)  # 대략 5kHz
            # band_energy = np.mean(stft[:, :, low_freq_idx:high_freq_idx])
            
            # 예시 5: 스펙트럴 센트로이드 (주파수 무게중심)
            # freq_axis = np.arange(freq_bins)
            # spectral_centroid = np.sum(stft * freq_axis[None, None, :]) / (np.sum(stft) + 1e-10)
            
            return float(mean_energy)
            
        except Exception as e:
            self.logger.error(f"Feature 계산 실패: {str(e)}")
            return 0.0
            
    def calculate_features_for_files(self, wav_files: List[Path], 
                                   audio_processor=None, 
                                   hdf5_manager=None) -> List[Dict[str, Any]]:
        """
        여러 파일에 대해 Feature 계산
        
        Args:
            wav_files: WAV 파일 경로 리스트
            audio_processor: AudioProcessor 인스턴스 (optional)
            hdf5_manager: HDF5Manager 인스턴스 (optional)
            
        Returns:
            List[Dict]: Feature 계산 결과 리스트
        """
        results = []
        
        # 필요시 모듈 import
        if audio_processor is None:
            from audio_processor import AudioProcessor
            audio_processor = AudioProcessor()
            
        if hdf5_manager is None:
            from hdf5_manager import HDF5Manager
            hdf5_manager = HDF5Manager()
            
        for i, wav_file in enumerate(wav_files):
            try:
                wav_path = Path(wav_file)
                
                self.logger.info(f"Processing {i+1}/{len(wav_files)}: {wav_path.name}")
                
                # STFT 데이터 로드 (새로운 HDF5 구조 사용)
                stft_data = None
                
                # 새로운 HDF5 구조에서 로드 시도 (폴더명.h5 파일에서 WAV 파일명 그룹)
                audio_data = hdf5_manager.load_audio_data(wav_path)
                
                if audio_data:
                    stft_data = audio_data['stft_data']
                    metadata = audio_data['metadata']
                else:
                    # HDF5에서 로드 실패시 WAV에서 직접 계산
                    audio_data = audio_processor.load_wav(wav_path)
                    if audio_data:
                        stft_data = audio_processor.compute_stft(audio_data['raw_data'])
                        metadata = audio_data['metadata']
                        
                if stft_data is not None:
                    # Feature 계산
                    feature_value = self.cal_feature(stft_data)
                    
                    # 결과 저장
                    result = {
                        'filename': wav_path.name,
                        'filepath': str(wav_path),
                        'feature_value': feature_value,
                        'num_channels': metadata.get('num_channels', 1),
                        'sample_rate': metadata.get('sampling_rate', 44100),
                        'duration': metadata.get('length_sec', 0),
                        'has_hdf5': audio_data is not None and 'stft_data' in audio_data
                    }
                    
                    results.append(result)
                    
                else:
                    self.logger.warning(f"STFT 데이터를 로드할 수 없음: {wav_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Feature 계산 실패 {wav_file}: {str(e)}")
                
        self.feature_results = results
        self.logger.info(f"Feature 계산 완료: {len(results)}개 파일")
        
        return results
        
    def show_histogram(self, feature_results: List[Dict[str, Any]] = None):
        """
        Feature 값들의 히스토그램 표시
        
        Args:
            feature_results: Feature 계산 결과 (None이면 self.feature_results 사용)
        """
        if feature_results is None:
            feature_results = self.feature_results
            
        if not feature_results:
            messagebox.showwarning("경고", "표시할 Feature 데이터가 없습니다.")
            return
            
        try:
            # Feature 값 추출
            feature_values = [result['feature_value'] for result in feature_results]
            filenames = [result['filename'] for result in feature_results]
            
            # 통계 계산
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # 히스토그램 플롯
            plt.figure(figsize=(12, 8))
            
            # 서브플롯 1: 히스토그램
            plt.subplot(2, 1, 1)
            n, bins, patches = plt.hist(feature_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.4f}')
            plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.4f}')
            
            plt.title(f'Feature Distribution (n={len(feature_values)})')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 통계 정보 텍스트
            stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 서브플롯 2: 파일별 Feature 값
            plt.subplot(2, 1, 2)
            x_pos = np.arange(len(feature_values))
            bars = plt.bar(x_pos, feature_values, alpha=0.7, color='lightgreen', edgecolor='black')
            
            # 평균선
            plt.axhline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            plt.title('Feature Values by File')
            plt.xlabel('File Index')
            plt.ylabel('Feature Value')
            plt.grid(True, alpha=0.3)
            
            # x축 레이블 (파일명이 너무 많으면 생략)
            if len(filenames) <= 20:
                plt.xticks(x_pos, [name[:15] + '...' if len(name) > 15 else name for name in filenames], 
                          rotation=45, ha='right')
            else:
                plt.xticks([])
                
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"히스토그램 표시 실패: {str(e)}")
            messagebox.showerror("오류", f"히스토그램 표시 중 오류 발생: {str(e)}")
            
    def save_results_to_csv(self, feature_results: List[Dict[str, Any]] = None, 
                           output_path: str = None):
        """
        Feature 계산 결과를 CSV 파일로 저장
        
        Args:
            feature_results: Feature 계산 결과 (None이면 self.feature_results 사용)
            output_path: 저장할 파일 경로 (None이면 파일 다이얼로그 표시)
        """
        if feature_results is None:
            feature_results = self.feature_results
            
        if not feature_results:
            messagebox.showwarning("경고", "저장할 Feature 데이터가 없습니다.")
            return False
            
        try:
            # 출력 경로 결정
            if output_path is None:
                output_path = filedialog.asksaveasfilename(
                    title="Feature 결과 저장",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                
            if not output_path:
                return False
                
            # DataFrame 생성 및 저장
            df = pd.DataFrame(feature_results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Feature 결과 저장 완료: {output_path}")
            messagebox.showinfo("완료", f"Feature 결과가 저장되었습니다:\n{output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV 저장 실패: {str(e)}")
            messagebox.showerror("오류", f"CSV 저장 중 오류 발생: {str(e)}")
            return False
            
    def get_feature_statistics(self, feature_results: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Feature 값들의 통계 정보 계산
        
        Args:
            feature_results: Feature 계산 결과 (None이면 self.feature_results 사용)
            
        Returns:
            Dict: 통계 정보
        """
        if feature_results is None:
            feature_results = self.feature_results
            
        if not feature_results:
            return {}
            
        feature_values = [result['feature_value'] for result in feature_results]
        
        stats = {
            'count': len(feature_values),
            'mean': np.mean(feature_values),
            'std': np.std(feature_values),
            'min': np.min(feature_values),
            'max': np.max(feature_values),
            'median': np.median(feature_values),
            'q25': np.percentile(feature_values, 25),
            'q75': np.percentile(feature_values, 75)
        }
        
        return stats
        
    def find_outliers(self, feature_results: List[Dict[str, Any]] = None, 
                     method: str = 'iqr', threshold: float = 1.5) -> List[Dict[str, Any]]:
        """
        이상치 파일 찾기
        
        Args:
            feature_results: Feature 계산 결과 (None이면 self.feature_results 사용)
            method: 이상치 탐지 방법 ('iqr' 또는 'zscore')
            threshold: 임계값
            
        Returns:
            List[Dict]: 이상치로 판단된 파일들
        """
        if feature_results is None:
            feature_results = self.feature_results
            
        if not feature_results:
            return []
            
        feature_values = np.array([result['feature_value'] for result in feature_results])
        outlier_indices = []
        
        if method == 'iqr':
            # IQR 방법
            q25 = np.percentile(feature_values, 25)
            q75 = np.percentile(feature_values, 75)
            iqr = q75 - q25
            
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            
            outlier_indices = np.where((feature_values < lower_bound) | 
                                     (feature_values > upper_bound))[0]
                                     
        elif method == 'zscore':
            # Z-score 방법
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            
            z_scores = np.abs((feature_values - mean_val) / std_val)
            outlier_indices = np.where(z_scores > threshold)[0]
            
        # 이상치 파일 정보 반환
        outliers = [feature_results[i] for i in outlier_indices]
        
        return outliers
        
    def compare_features(self, feature_results1: List[Dict[str, Any]], 
                        feature_results2: List[Dict[str, Any]], 
                        label1: str = "Group 1", label2: str = "Group 2"):
        """
        두 그룹의 Feature 값 비교 시각화
        
        Args:
            feature_results1: 첫 번째 그룹의 Feature 결과
            feature_results2: 두 번째 그룹의 Feature 결과
            label1: 첫 번째 그룹 레이블
            label2: 두 번째 그룹 레이블
        """
        try:
            values1 = [result['feature_value'] for result in feature_results1]
            values2 = [result['feature_value'] for result in feature_results2]
            
            plt.figure(figsize=(12, 6))
            
            # 서브플롯 1: 히스토그램 비교
            plt.subplot(1, 2, 1)
            plt.hist(values1, bins=15, alpha=0.7, label=label1, color='skyblue')
            plt.hist(values2, bins=15, alpha=0.7, label=label2, color='lightcoral')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.title('Feature Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 서브플롯 2: 박스플롯 비교
            plt.subplot(1, 2, 2)
            plt.boxplot([values1, values2], labels=[label1, label2])
            plt.ylabel('Feature Value')
            plt.title('Feature Value Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Feature 비교 시각화 실패: {str(e)}")
            messagebox.showerror("오류", f"Feature 비교 중 오류 발생: {str(e)}")