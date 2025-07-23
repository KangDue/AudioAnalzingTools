# Audio Analysis & Preprocessing Tool

**Python + customtkinter 기반 오디오 분석 및 전처리 도구**

## 프로젝트 개요

이 도구는 WAV 파일을 전처리하여 HDF5 형식으로 저장하고, 오디오 데이터의 Waveform, Spectrum, Spectrogram을 시각화하며, 사용자 정의 Feature를 계산할 수 있는 GUI 애플리케이션입니다.

## 주요 기능

### 1. 폴더 선택 & 파일 리스트 표시
- 폴더 선택을 통해 WAV 파일 자동 스캔
- 파일 크기 및 HDF5 변환 상태 표시
- 파일 선택을 통한 개별 분석

### 2. 전처리 (WAV → HDF5 변환)
- WAV 파일을 로드하여 다음 데이터를 계산 및 저장:
  - **Raw Data**: 원본 오디오 데이터 (float32)
  - **STFT**: Short-Time Fourier Transform magnitude (float32)
  - **Spectrum**: 전체 FFT magnitude (float32)
  - **Metadata**: 샘플링 레이트, 채널 수, 길이 등

### 3. 시각화
- **Waveform**: 시간 영역 파형 표시
- **Spectrum**: 주파수 영역 스펙트럼 (로그 스케일 옵션)
- **Spectrogram**: STFT 기반 스펙트로그램 (컬러맵 선택 가능)

### 4. Feature 계산
- 사용자 정의 Feature 함수를 통한 특성 추출
- 모든 파일에 대한 Feature 값 계산
- 히스토그램을 통한 분포 시각화
- CSV 형태로 결과 저장

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

```bash
python main.py
```

## 사용법

### 1. 기본 워크플로우

1. **폴더 선택**: "폴더 선택" 버튼을 클릭하여 WAV 파일이 있는 폴더 선택
2. **전처리**: "전처리 (WAV → HDF5)" 버튼을 클릭하여 모든 WAV 파일을 HDF5로 변환
3. **파일 분석**: 좌측 파일 리스트에서 파일을 선택하여 시각화 확인
4. **Feature 계산**: "Cal Features" 버튼을 클릭하여 모든 파일의 Feature 계산 및 히스토그램 표시

### 2. HDF5 데이터 구조

```
root
├── raw_data       # float32, [channels, samples]
├── stft           # float32, [channels, frames, freq_bins]  (magnitude)
├── spectrum       # float32, [channels, freq_bins]          (magnitude)
└── meta           # Attributes: sampling_rate, num_channels, length_sec 등
```

### 3. Feature 함수 커스터마이징

`feature_calculator.py`의 `cal_feature` 함수를 수정하여 원하는 Feature를 계산할 수 있습니다:

```python
def cal_feature(self, stft: np.ndarray) -> float:
    """
    사용자 정의 Feature 계산 함수
    
    Args:
        stft: np.ndarray, shape [channels, frames, freq_bins] - STFT magnitude
        
    Returns:
        float: 계산된 Feature 값
    """
    # 예시: 평균 에너지
    return np.mean(stft)
    
    # 다른 예시들:
    # return np.max(stft)  # 최대 에너지
    # return np.std(stft)  # 에너지 표준편차
    # return np.mean(stft[:, :, 10:50])  # 특정 주파수 대역 에너지
```

## 프로젝트 구조

```
simple_wav_view/
├── main.py                 # 메인 GUI 애플리케이션
├── audio_processor.py      # 오디오 처리 (WAV 로드, STFT, Spectrum)
├── hdf5_manager.py         # HDF5 데이터 저장/로드
├── gui_components.py       # GUI 컴포넌트 (파일 리스트, 시각화)
├── feature_calculator.py   # Feature 계산 및 히스토그램
├── requirements.txt        # 의존성 목록
├── README.md              # 프로젝트 설명서
├── work_log.txt           # 개발 작업일지
└── project_brief.md       # 프로젝트 명세서
```

## 기술 스택

- **GUI Framework**: customtkinter
- **오디오 처리**: soundfile, numpy, scipy.signal
- **데이터 저장**: h5py (HDF5)
- **시각화**: matplotlib
- **데이터 분석**: pandas

## 주요 특징

### 1. 효율적인 데이터 관리
- HDF5 형식을 통한 압축 저장
- 한 번 계산된 STFT/Spectrum 데이터 재사용
- 메타데이터와 함께 통합 관리

### 2. 실시간 시각화
- 파일 선택 시 즉시 로드 및 시각화
- 다중 채널 지원
- 인터랙티브 플롯 (로그 스케일, 컬러맵 변경)

### 3. 확장 가능한 Feature 계산
- 사용자 정의 Feature 함수
- 배치 처리 지원
- 통계 분석 및 시각화

## 사용 예시

### 1. 음성 데이터 분석
- 음성 파일들의 에너지 분포 분석
- 스펙트럴 특성 비교
- 이상치 탐지

### 2. 음악 분석
- 장르별 스펙트럼 특성 분석
- 템포 및 리듬 패턴 분석
- 주파수 대역별 에너지 분포

### 3. 환경음 분석
- 소음 레벨 모니터링
- 특정 주파수 성분 추출
- 시간대별 음향 특성 변화

## 문제 해결

### 1. 메모리 부족
- 큰 파일의 경우 STFT 윈도우 크기 조정
- 배치 크기 줄이기
- HDF5 압축 옵션 활용

### 2. 처리 속도 개선
- HDF5 파일 우선 사용
- 멀티프로세싱 적용 (향후 개선)
- 필요한 채널만 처리

### 3. GUI 응답성
- 백그라운드 스레드에서 전처리 실행
- 진행률 표시
- 비동기 파일 로드

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다. GitHub Issues를 통해 문의해 주세요.