#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 Data Management Module
HDF5 파일 저장/로드 담당
"""

import h5py
import numpy as np
from pathlib import Path
import logging
import json


class HDF5Manager:
    """HDF5 데이터 관리 클래스"""
    
    def __init__(self):
        """
        초기화
        """
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # HDF5 데이터 구조 정의
        self.data_structure = {
            'raw_data': 'float32',      # [channels, samples]
            'stft': 'float32',          # [channels, frames, freq_bins] - magnitude only
            'spectrum': 'float32',      # [channels, freq_bins] - magnitude only
            'meta': 'attributes'        # metadata as attributes
        }
        
    def save_audio_data(self, file_path, raw_data, stft_data, spectrum_data, metadata):
        """
        오디오 데이터를 HDF5 파일로 저장
        
        Args:
            file_path: 저장할 HDF5 파일 경로 (WAV 파일 경로)
            raw_data: np.ndarray, shape [channels, samples]
            stft_data: np.ndarray, shape [channels, frames, freq_bins]
            spectrum_data: np.ndarray, shape [channels, freq_bins]
            metadata: dict, 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            file_path = Path(file_path)
            folder_name = file_path.parent.name
            file_name = file_path.stem  # 확장자 제거한 파일명
            
            # HDF5 파일 경로: 폴더명.h5
            h5_file_path = file_path.parent / f"{folder_name}.h5"
            
            # 파일이 존재하면 append 모드, 없으면 write 모드
            mode = 'a' if h5_file_path.exists() else 'w'
            
            with h5py.File(str(h5_file_path), mode) as h5f:
                # 파일명으로 그룹 생성 (이미 존재하면 삭제 후 재생성)
                if file_name in h5f:
                    del h5f[file_name]
                    
                file_group = h5f.create_group(file_name)
                
                # Raw data 저장
                file_group.create_dataset(
                    'raw_data', 
                    data=raw_data.astype(np.float32),
                    compression='gzip',
                    compression_opts=6
                )
                
                # STFT data 저장 (magnitude only)
                file_group.create_dataset(
                    'stft', 
                    data=stft_data.astype(np.float32),
                    compression='gzip',
                    compression_opts=6
                )
                
                # Spectrum data 저장 (magnitude only)
                file_group.create_dataset(
                    'spectrum', 
                    data=spectrum_data.astype(np.float32),
                    compression='gzip',
                    compression_opts=6
                )
                
                # 메타데이터를 attributes로 저장
                meta_group = file_group.create_group('meta')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    else:
                        # 복잡한 데이터는 JSON 문자열로 저장
                        meta_group.attrs[key] = json.dumps(value)
                        
                # 저장 시간 기록
                import datetime
                meta_group.attrs['saved_at'] = datetime.datetime.now().isoformat()
                meta_group.attrs['hdf5_version'] = h5py.version.version
                
            self.logger.info(f"HDF5 저장 완료: {h5_file_path.name} -> {file_name} 그룹")
            return True
            
        except Exception as e:
            self.logger.error(f"HDF5 저장 실패 {file_path}: {str(e)}")
            return False
            
    def load_audio_data(self, file_path):
        """
        HDF5 파일에서 오디오 데이터 로드
        
        Args:
            file_path: WAV 파일 경로 또는 HDF5 파일 경로
            
        Returns:
            dict: {
                'raw_data': np.ndarray,
                'stft_data': np.ndarray,
                'spectrum_data': np.ndarray,
                'metadata': dict
            } 또는 None (실패시)
        """
        try:
            file_path = Path(file_path)
            
            # WAV 파일 경로인 경우 HDF5 파일 경로로 변환
            if file_path.suffix.lower() == '.wav':
                folder_name = file_path.parent.name
                file_name = file_path.stem
                h5_file_path = file_path.parent / f"{folder_name}.h5"
            else:
                # 이미 HDF5 파일 경로인 경우
                h5_file_path = file_path
                file_name = None  # 첫 번째 그룹 사용
            
            if not h5_file_path.exists():
                self.logger.error(f"HDF5 파일이 존재하지 않음: {h5_file_path}")
                return None
                
            with h5py.File(str(h5_file_path), 'r') as h5f:
                # 파일명 그룹 찾기
                if file_name and file_name in h5f:
                    file_group = h5f[file_name]
                elif len(h5f.keys()) > 0:
                    # 파일명이 없으면 첫 번째 그룹 사용
                    first_group_name = list(h5f.keys())[0]
                    file_group = h5f[first_group_name]
                    file_name = first_group_name
                else:
                    # 기존 구조 호환성 유지 (루트에 직접 데이터가 있는 경우)
                    if 'raw_data' in h5f:
                        raw_data = h5f['raw_data'][:].astype(np.float32)
                        stft_data = h5f['stft'][:].astype(np.float32)
                        spectrum_data = h5f['spectrum'][:].astype(np.float32)
                        
                        metadata = {}
                        if 'meta' in h5f:
                            meta_group = h5f['meta']
                            for key in meta_group.attrs.keys():
                                value = meta_group.attrs[key]
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                                
                                if isinstance(value, str) and value.startswith(('{', '[')):
                                    try:
                                        value = json.loads(value)
                                    except json.JSONDecodeError:
                                        pass
                                        
                                metadata[key] = value
                                
                        self.logger.info(f"HDF5 로드 완료 (기존 구조): {h5_file_path.name}")
                        return {
                            'raw_data': raw_data,
                            'stft_data': stft_data,
                            'spectrum_data': spectrum_data,
                            'metadata': metadata
                        }
                    else:
                        self.logger.error(f"HDF5 파일에 유효한 데이터가 없음: {h5_file_path}")
                        return None
                
                # 새로운 구조에서 데이터 로드
                raw_data = file_group['raw_data'][:].astype(np.float32)
                stft_data = file_group['stft'][:].astype(np.float32)
                spectrum_data = file_group['spectrum'][:].astype(np.float32)
                
                # 메타데이터 로드
                metadata = {}
                if 'meta' in file_group:
                    meta_group = file_group['meta']
                    for key in meta_group.attrs.keys():
                        value = meta_group.attrs[key]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        
                        # JSON 문자열인지 확인하여 파싱
                        if isinstance(value, str) and value.startswith(('{', '[')):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                pass  # JSON이 아니면 그대로 사용
                                
                        metadata[key] = value
                        
            self.logger.info(f"HDF5 로드 완료: {h5_file_path.name} -> {file_name} 그룹")
            
            return {
                'raw_data': raw_data,
                'stft_data': stft_data,
                'spectrum_data': spectrum_data,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"HDF5 로드 실패 {file_path}: {str(e)}")
            return None
            
    def check_hdf5_integrity(self, file_path):
        """
        HDF5 파일 무결성 검사
        
        Args:
            file_path: HDF5 파일 경로
            
        Returns:
            dict: 검사 결과
        """
        try:
            file_path = Path(file_path)
            result = {
                'valid': False,
                'has_raw_data': False,
                'has_stft': False,
                'has_spectrum': False,
                'has_metadata': False,
                'shapes': {},
                'errors': []
            }
            
            if not file_path.exists():
                result['errors'].append("File does not exist")
                return result
                
            with h5py.File(str(file_path), 'r') as h5f:
                # 폴더명 그룹 찾기
                folder_name = file_path.parent.name
                
                # 새로운 구조 확인
                if folder_name in h5f:
                    folder_group = h5f[folder_name]
                    # 필수 데이터셋 확인
                    required_datasets = ['raw_data', 'stft', 'spectrum']
                    
                    for dataset_name in required_datasets:
                        if dataset_name in folder_group:
                            result[f'has_{dataset_name}'] = True
                            result['shapes'][dataset_name] = folder_group[dataset_name].shape
                        else:
                            result['errors'].append(f"Missing dataset: {dataset_name}")
                            
                    # 메타데이터 확인
                    if 'meta' in folder_group:
                        result['has_metadata'] = True
                        result['metadata_keys'] = list(folder_group['meta'].attrs.keys())
                    else:
                        result['errors'].append("Missing metadata group")
                else:
                    # 기존 구조 호환성 확인
                    required_datasets = ['raw_data', 'stft', 'spectrum']
                    
                    for dataset_name in required_datasets:
                        if dataset_name in h5f:
                            result[f'has_{dataset_name}'] = True
                            result['shapes'][dataset_name] = h5f[dataset_name].shape
                        else:
                            result['errors'].append(f"Missing dataset: {dataset_name}")
                            
                    # 메타데이터 확인
                    if 'meta' in h5f:
                        result['has_metadata'] = True
                        result['metadata_keys'] = list(h5f['meta'].attrs.keys())
                    else:
                        result['errors'].append("Missing metadata group")
                    
                # 데이터 형태 일관성 확인
                if result['has_raw_data'] and result['has_stft'] and result['has_spectrum']:
                    raw_shape = result['shapes']['raw_data']
                    stft_shape = result['shapes']['stft']
                    spectrum_shape = result['shapes']['spectrum']
                    
                    # 채널 수 일치 확인
                    if raw_shape[0] == stft_shape[0] == spectrum_shape[0]:
                        result['channels_consistent'] = True
                    else:
                        result['errors'].append("Channel count mismatch between datasets")
                        result['channels_consistent'] = False
                        
            # 전체 유효성 판단
            result['valid'] = (len(result['errors']) == 0 and 
                             result['has_raw_data'] and 
                             result['has_stft'] and 
                             result['has_spectrum'] and 
                             result['has_metadata'])
                             
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Exception during integrity check: {str(e)}"]
            }
            
    def get_hdf5_info(self, file_path):
        """
        HDF5 파일 정보 요약
        
        Args:
            file_path: HDF5 파일 경로
            
        Returns:
            str: 파일 정보 문자열
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return f"File not found: {file_path}"
                
            integrity = self.check_hdf5_integrity(file_path)
            
            if not integrity['valid']:
                return f"Invalid HDF5 file: {', '.join(integrity['errors'])}"
                
            # 파일 크기
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # 데이터 형태 정보
            shapes = integrity['shapes']
            
            info = (
                f"HDF5 File: {file_path.name}\n"
                f"File Size: {file_size_mb:.2f} MB\n"
                f"Raw Data: {shapes.get('raw_data', 'N/A')}\n"
                f"STFT Data: {shapes.get('stft', 'N/A')}\n"
                f"Spectrum Data: {shapes.get('spectrum', 'N/A')}\n"
                f"Metadata Keys: {len(integrity.get('metadata_keys', []))}\n"
                f"Valid: {integrity['valid']}"
            )
            
            return info
            
        except Exception as e:
            return f"Error getting HDF5 info: {str(e)}"
            
    def list_hdf5_files(self, directory):
        """
        디렉토리에서 HDF5 파일 목록 조회
        
        Args:
            directory: 검색할 디렉토리 경로
            
        Returns:
            list: HDF5 파일 경로 리스트
        """
        try:
            directory = Path(directory)
            h5_files = list(directory.glob("*.h5"))
            h5_files.extend(list(directory.glob("*.hdf5")))
            
            return sorted(h5_files)
            
        except Exception as e:
            self.logger.error(f"HDF5 파일 목록 조회 실패: {str(e)}")
            return []
            
    def convert_wav_to_hdf5_batch(self, wav_files, audio_processor, progress_callback=None):
        """
        WAV 파일들을 일괄 HDF5로 변환
        
        Args:
            wav_files: WAV 파일 경로 리스트
            audio_processor: AudioProcessor 인스턴스
            progress_callback: 진행률 콜백 함수 (optional)
            
        Returns:
            dict: 변환 결과 통계
        """
        results = {
            'total': len(wav_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        # 폴더별로 그룹화
        folders = {}
        for wav_file in wav_files:
            wav_path = Path(wav_file)
            folder_name = wav_path.parent.name
            if folder_name not in folders:
                folders[folder_name] = []
            folders[folder_name].append(wav_path)
        
        processed_count = 0
        
        for folder_name, folder_files in folders.items():
            try:
                # 폴더명.h5 파일 경로
                h5_file_path = folder_files[0].parent / f"{folder_name}.h5"
                
                # 이미 존재하는 파일인지 확인
                existing_groups = set()
                if h5_file_path.exists():
                    try:
                        with h5py.File(str(h5_file_path), 'r') as h5f:
                            existing_groups = set(h5f.keys())
                    except:
                        pass
                
                for wav_path in folder_files:
                    processed_count += 1
                    file_name = wav_path.stem
                    
                    try:
                        # 이미 처리된 파일이면 스킵
                        if file_name in existing_groups:
                            results['skipped'] += 1
                            if progress_callback:
                                progress_callback(processed_count, results['total'], f"Skipped: {wav_path.name}")
                            continue
                            
                        # WAV 로드
                        audio_data = audio_processor.load_wav(wav_path)
                        if audio_data is None:
                            results['failed'] += 1
                            results['errors'].append(f"Failed to load: {wav_path.name}")
                            continue
                            
                        # STFT 및 Spectrum 계산
                        stft_data = audio_processor.compute_stft(audio_data['raw_data'])
                        spectrum_data = audio_processor.compute_spectrum(audio_data['raw_data'])
                        
                        if stft_data is None or spectrum_data is None:
                            results['failed'] += 1
                            results['errors'].append(f"Failed to compute features: {wav_path.name}")
                            continue
                            
                        # HDF5 저장 (WAV 파일 경로를 전달)
                        success = self.save_audio_data(
                            wav_path,  # WAV 파일 경로 전달
                            audio_data['raw_data'],
                            stft_data,
                            spectrum_data,
                            audio_data['metadata']
                        )
                        
                        if success:
                            results['success'] += 1
                            existing_groups.add(file_name)  # 성공한 파일 추가
                        else:
                            results['failed'] += 1
                            results['errors'].append(f"Failed to save HDF5: {wav_path.name}")
                            
                        # 진행률 콜백
                        if progress_callback:
                            status = "Success" if success else "Failed"
                            progress_callback(processed_count, results['total'], f"{status}: {wav_path.name}")
                            
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"Exception processing {wav_path.name}: {str(e)}")
                        
            except Exception as e:
                # 폴더 전체 처리 실패
                for wav_path in folder_files:
                    processed_count += 1
                    results['failed'] += 1
                    results['errors'].append(f"Exception processing folder {folder_name}: {str(e)}")
                    if progress_callback:
                        progress_callback(processed_count, results['total'], f"Failed: {wav_path.name}")
                
        return results