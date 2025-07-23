# **최종 업데이트된 프로젝트 명세서**

**프로젝트명:** Audio Analysis & Preprocessing Tool (Python + customtkinter)
**UI Framework:** customtkinter
**핵심 라이브러리:**

* `soundfile` (WAV 로드)
* `numpy`, `scipy.signal` (FFT, STFT, Spectrum 계산)
* `h5py` (HDF5 저장/로드)
* `matplotlib` (FigureCanvasTkAgg로 시각화)

---

## 1. 프로그램 개요

폴더 내 WAV 파일을 전처리하여 **HDF5** 파일로 저장하고,
파일별 \*\*Waveform, Spectrum, Spectrogram(STFT)\*\*을 시각화/분석하는 **Python GUI 툴**.
HDF5에는 **STFT magnitude (abs 값, float32)**, **Spectrum (FFT)**, **Raw WAV**를 모두 저장하여, 로드 시 모든 데이터를 즉시 활용할 수 있게 한다.

---

## 2. 주요 기능 (최종 반영)

### (1) 폴더 선택 & 파일 리스트 표시

* \[폴더 선택] 버튼 → 디렉토리 선택 → WAV 파일 목록 표시 (메타데이터 포함).

---

### (2) 전처리 (HDF5 변환)

* \[전처리] 버튼:

  1. WAV 파일 로드 (soundfile)
  2. 채널별 STFT 계산 (`scipy.signal.stft`)
  3. STFT는 **복소수(complex)** 대신 \*\*magnitude (abs)\*\*를 계산하여 **float32**로 저장
  4. 전체 Spectrum 계산 (채널별 FFT)
  5. 메타데이터 추출
  6. `<원본파일명>.h5`로 저장 (같은 폴더)

HDF5 저장 구조 (최종):

```text
root
├── raw_data       # float32, [channels, samples]
├── stft           # float32, [channels, frames, freq_bins]  (abs 값만 저장)
├── spectrum       # float32, [channels, freq_bins]          (전체 FFT magnitude)
└── meta           # Attributes: sampling_rate, bit_depth, num_channels, length_sec 등
```

---

### (3) 파일 선택 시 로드 & 시각화

* `.h5` 파일이 있으면: `raw_data`, `stft`, `spectrum` 모두 로드
* 없으면 WAV 로드 후 STFT/Spectrum 즉시 계산
* 시각화 (탭으로 분리):

  * **Waveform (시간영역)**
  * **Spectrum (전체 FFT magnitude, 로그 스케일 가능)**
  * **Spectrogram (STFT magnitude 히트맵)**

---

### (4) 사용자 정의 Feature 계산

* 사용자 정의 함수:

  ```python
  def cal_feature(stft: np.ndarray) -> float:
      # stft shape: (channels, frames, freq_bins)  - magnitude 값
      return np.mean(stft)  # 예시: 평균 에너지
  ```
* \[Cal Features] 버튼:

  * 모든 파일의 STFT magnitude를 순회하며 `cal_feature` 실행
  * 결과 히스토그램 시각화 (Matplotlib)
  * CSV 저장 기능 제공

---

## 3. Spectrum 계산 기본값

* FFT 길이: **파일 전체 길이 또는 가장 가까운 2의 거듭제곱 길이 (pad)**
* Magnitude만 float32로 저장 (로그 변환은 시각화 단계에서 적용 가능).

---

## 4. 처리 흐름 (최종)

```
[폴더 선택]
     ↓
WAV 파일 목록 표시
     ↓
[전처리]
     ↓
WAV 로드 → STFT (abs) → Spectrum (abs) → HDF5 저장
     ↓
파일 선택 시
     ↓
(HDF5 로드 or WAV 로드) → raw_data + stft(abs) + spectrum(abs)
     ↓
Waveform / Spectrum / Spectrogram 표시
     ↓
[Cal Features] (STFT magnitude 입력 → 결과 계산 → 히스토그램)
```

---

### **다음 단계 선택**

이제 이 명세를 바탕으로:

1. **폴더 구조 + 초기 모듈 스캐폴드 (`main.py`, `stft_utils.py`, `hdf5_manager.py` 등)**
2. **실행 가능한 `main.py` GUI 뼈대 (폴더 선택, 리스트뷰, 빈 플롯만 표시)**
3. **STFT + Spectrum + HDF5 저장 모듈 코드**
4. **3개 모두 통합한 초기 버전 (동작 가능한 최소 기능 앱)**

어떤 걸 먼저 만들어줄까?
아니면 \*\*4번 (동작 가능한 기본 앱)\*\*으로 바로 갈까?
