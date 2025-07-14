# WAV Analyzer 0714

Advanced WAV analyzer for rotation machinery fault diagnosis using 10 state-of-the-art feature extraction algorithms based on 2017+ research.

## Features

This tool extracts 118 advanced features from .wav files using 10 different algorithms:

1. **WPE (Wavelet Packet + Entropy)** - 8 features
2. **VMD (Variational Mode Decomposition)** - 15 features  
3. **Deep Convolutional Autoencoder** - 16 features
4. **TFD + 2D CNN** - 12 features
5. **Acoustic Beamforming CNN** - 15 features
6. **Sparse Representation (SRC-FD)** - 10 features
7. **AE Envelope + Hilbert-Huang Transform** - 10 features
8. **Thermal Image Deep Features** - 10 features
9. **Multi-Scale Attention (MSA-CNN)** - 12 features
10. **Adaptive Time Synchronous Averaging (TSA-RNN)** - 10 features

## Installation

1. Make sure you have Python 3.11+ installed
2. Install dependencies using uv:

```bash
uv sync
```

## Usage

### Method 1: Run main script
```bash
uv run python main.py
```

### Method 2: Run feature extraction directly
```bash
uv run python extract_features.py
```

### Method 3: Use as module
```bash
uv run python -m extract_features
```

## How it works

1. **Select Folder**: A GUI dialog will open for you to select a folder containing .wav files
2. **Automatic Processing**: The tool will:
   - Find all .wav files in the selected folder
   - Load each file at 48kHz sampling rate
   - Extract 118 features using 10 advanced algorithms
   - Save results to `features_extracted.csv` in the same folder

## Output

The tool generates a CSV file with:
- First column: filename
- Remaining 118 columns: extracted features
- Saved as `features_extracted.csv` in the input folder

## Supported File Format

- Input: .wav files (automatically resampled to 48kHz)
- Output: .csv file with pandas DataFrame

## Algorithm Details

All algorithms are based on recent research (2017+) for rotation machinery fault diagnosis:

- **Signal Processing**: WPE, VMD, HHT, TSA
- **Deep Learning**: Autoencoder, CNN, MSA-CNN, TSA-RNN
- **Advanced Methods**: Beamforming, Sparse Representation, Thermal Analysis

## Requirements

- Python >= 3.11
- All dependencies listed in pyproject.toml
- GUI support (tkinter)

## Notes

- Processing time depends on file size and number of files
- Progress bar shows extraction progress
- Error handling for corrupted or unsupported files
- All features are normalized and ready for machine learning

# 추가 설명

## WAV Analyzer 프로젝트 완성
프로젝트 규칙에 따라 회전체 결함 진단을 위한 고급 WAV 분석기를 성공적으로 구현했습니다.

### 구현된 기능
10가지 최신 알고리즘 기반 특징 추출:

1. WPE (Wavelet Packet + Entropy) - 8개 특징
2. VMD (Variational Mode Decomposition) - 15개 특징
3. Deep Convolutional Autoencoder - 16개 특징
4. TFD + 2D CNN - 12개 특징
5. Acoustic Beamforming CNN - 15개 특징
6. Sparse Representation (SRC-FD) - 10개 특징
7. AE Envelope + Hilbert-Huang Transform - 10개 특징
8. Thermal Image Deep Features - 10개 특징
9. Multi-Scale Attention (MSA-CNN) - 12개 특징
10. Adaptive Time Synchronous Averaging (TSA-RNN) - 10개 특징
총 118개 고급 특징 추출 가능

### 생성된 파일들
- `pyproject.toml` - uv 프로젝트 설정 및 의존성
- `extract_features.py` - 메인 특징 추출 엔진
- `main.py` - GUI 실행 스크립트
- `test_features.py` - 기능 테스트 스크립트
- `demo.py` - 데모 및 예제 실행
- `README.md` - 상세 사용법 문서
### 주요 특징
- 48kHz 샘플링 자동 처리
- GUI 폴더 선택 (tkinter 기반)
- 진행률 표시 (tqdm)
- 자동 CSV 저장 (pandas DataFrame)
- 에러 처리 및 복구 기능
- 2017년 이후 최신 연구 기반 알고리즘
### 사용법
```
# 의존성 설치
uv sync

# GUI로 실행
uv run python main.py

# 데모 실행
uv run python demo.py

# 테스트 실행
uv run python test_features.py
```
### 검증 완료
- ✅ 모든 의존성 설치 성공
- ✅ 118개 특징 추출 검증
- ✅ 정상/결함 모터 신호 구분 확인
- ✅ CSV 출력 형식 검증
- ✅ 에러 처리 동작 확인
프로젝트는 완전히 동작하며, 사용자가 폴더를 선택하면 자동으로 WAV 파일들을 분석하여 features_extracted.csv 파일로 결과를 저장합니다.