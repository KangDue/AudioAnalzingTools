아래는 네가 요청한 **파이썬 기반 오디오 포커싱(Audio Focusing) 시뮬레이션 프로그램** 명세서야.
\*\*원형 마이크 어레이 + 평면 타겟 (70x70 픽셀)\*\*을 가정하고, **FD(Filter-and-Sum Delay) 기반의 Beamforming/포커싱 알고리즘**으로 효율성을 확보한 구조로 작성했어.

---

# **프로젝트 명세서**

**프로젝트명:** Audio Focusing & Visualization Simulator
**언어/플랫폼:** Python
**UI (선택):** customtkinter (결과 시각화) 또는 Matplotlib Interactive
**핵심 알고리즘:** Fractional Delay Filter-and-Sum Beamforming (FD-FAS) - 관련하여 최신 효율적인 알고리즘 조사한뒤 구현하도록
**주요 목적:**

* 원형 배열된 마이크 어레이로 특정 평면(70x70 grid)의 음장 포커싱/시각화
* 효율적인 주파수영역 기반 포커싱 알고리즘 적용
* 시뮬레이션 데이터 기반으로 음장 세기(에너지) 맵 생성

---

## 1. 시뮬레이션 개요

1. **마이크 어레이 구성**

   * 원형 배열 (반지름 R, 예: 0.1m)
   * 마이크 개수: 사용자가 지정 (예: 8\~16개)
   * 2D 평면 상 원점 중심 배치 (각 마이크 각도 균일 분포)

2. **타겟 평면**

   * 크기: 70x70 grid
   * 평면 좌표계: z=Z\_target (마이크 배열에서 수직으로 거리 떨어져 있음, 예: 1m)
   * 각 픽셀에서 음압 레벨 또는 에너지를 계산하여 시각화

3. **신호 모델**

   * 소스: 1\~N개의 음원 (예: 단일 소스 or 랜덤 위치)
   * 음원 신호: 단순 사인파, 임펄스, 또는 화이트 노이즈 (사용자 선택)
   * 전파 모델: Free-field 전파, 지연(delay)과 감쇠만 고려 (회절/반사 무시)

4. **포커싱 알고리즘**

   * **FD Filter-and-Sum Beamforming (FD-FAS)**

     * 시간영역 delay-and-sum 대신, STFT 후 주파수별 위상 보정 & 가중치 곱
     * FFT 기반 필터링으로 연산 효율화 (O(N log N))
     * 70x70 grid 각 포인트마다 steering vector를 적용, 에너지 맵 계산

---

## 2. 알고리즘 상세 - 이건 하나의 예시일뿐이고, 자료 조사를 통해 효율적인 알고리즘 구현할 것

1. **STFT (Short-Time Fourier Transform)**

   * 각 마이크 입력에 대해 STFT 수행
   * 윈도우 크기와 hop size 설정 가능 (기본: 1024 FFT, hop=512)

2. **Steering Vector 계산**

   * grid의 각 포인트에 대해, 각 마이크까지의 거리 `d_m(x,y)` 계산
   * 각 주파수 bin에서 위상 보정 인자:

     $$
     w_m(f, x, y) = e^{-j 2 \pi f d_m / c}
     $$

     (c = 음속, 343 m/s)

3. **Frequency-Domain Filter-and-Sum**

   * 각 포인트에 대해:

     $$
     P(f,x,y) = \frac{1}{M} \sum_{m=1}^{M} X_m(f) w_m(f,x,y)
     $$
   * IFFT 후 시간영역 신호 또는 주파수 에너지:

     $$
     E(x,y) = \sum_{f} |P(f,x,y)|^2
     $$

4. **시각화 (Visualization)**

   * 계산된 70x70 grid의 `E(x,y)`를 2D heatmap으로 표시
   * Matplotlib `imshow` 또는 `pcolormesh` 사용
   * colormap, 로그 스케일 옵션 지원

---

## 3. 주요 기능

1. **시뮬레이션 설정**

   * 마이크 개수 (8\~16)
   * 원형 배열 반지름 (0.05\~0.2m)
   * 타겟 평면 거리 (0.5\~2.0m)
   * FFT 파라미터 (윈도우, hop)
   * 음원 위치, 음원 신호 선택

2. **연산 효율화**

   * FFT 기반으로 주파수별 steering 적용
   * 벡터화 연산 (NumPy)
   * 필요 시 multiprocessing (CPU 코어 분산)

3. **결과 출력**

   * 70x70 음장 에너지 히트맵
   * 선택 포인트의 파형/스펙트럼 보기
   * 결과를 HDF5로 저장 (`/energy_map`, `/waveforms`, `/meta`)

---

## 4. 내부 모듈 설계

* `mic_array.py`

  * 원형 배열 좌표 생성, steering vector 계산

* `signal_model.py`

  * 가상 소스 생성 (사인파, 노이즈 등)
  * 각 마이크 입력 시뮬레이션 (지연/감쇠 적용)

* `fd_beamforming.py`

  * STFT 수행
  * FD Filter-and-Sum 알고리즘 구현
  * 70x70 grid 에너지 맵 반환

* `plot_utils.py`

  * Heatmap, 파형, 스펙트럼 표시 함수

* `main.py`

  * 파라미터 입력, 연산 실행, 시각화

---

## 5. 처리 흐름

```
[마이크 배열 설정] + [소스 정의]
       ↓
가상 마이크 신호 시뮬레이션 (딜레이/감쇠)
       ↓
각 마이크 STFT
       ↓
70x70 grid의 각 포인트별 steering 적용 (FD Filter-and-Sum)
       ↓
E(x,y) 에너지 맵 계산
       ↓
Heatmap으로 시각화 (+ 선택 포인트 파형/스펙트럼)
```

---

## 6. 확장 아이디어

* 실시간 오디오 스트림 입력 (PyAudio)
* Delay-and-Sum vs FD Filter-and-Sum 성능 비교
* Beam pattern (특정 포인트 포커싱) 시뮬레이션 추가
* GPU 가속 (CuPy)로 grid 연산 속도 개선

---

### 다음 단계 중 뭘 먼저 만들어줄까?

1. **프로젝트 폴더 구조 + 빈 모듈 스캐폴드 (`mic_array.py`, `fd_beamforming.py` 등)**
2. **주요 알고리즘(FD Filter-and-Sum) 핵심 코드 예시 (STFT → 포커싱 → Heatmap)**
3. **Matplotlib 기반 인터랙티브 시뮬레이션 기본 버전 (`main.py` 단독 실행)**
4. **3개 모두 합친 동작 가능한 초기 버전**

어느 방식으로 시작할래?
그리고 \*\*마이크 개수(예: 8개)와 FFT 파라미터(1024/512)\*\*를 기본값으로 할까? 아니면 직접 지정할 수 있게 할까?
