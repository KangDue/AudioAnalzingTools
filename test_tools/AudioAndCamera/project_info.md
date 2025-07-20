아래는 **기존 명세서**를 **시리얼 장치(마이크어레이) 또는 USB 웹캠이 없을 경우 자동으로 Fake 장치로 대체**하도록 수정한 버전입니다.

---

## **프로젝트 명세서 (수정판): "동시 녹화 PySerial + 카메라 (장치 유무 자동 감지, Fake 지원)"**

### **1. 프로젝트 개요**

* USB/Serial 기반 마이크 어레이와 USB 웹캠을 **자동으로 감지**.
* 실제 장치가 연결되어 있지 않으면 **Fake 장치로 자동 대체**하여 프로그램이 항상 동작.
* 오디오/비디오를 동기화 녹화 후 STFT, FFT, Waveform, Heatmap Overlay 분석.
* **PyQt5 GUI + PySerial + PyQtGraph + OpenCV 기반**.

---

### **2. Fake 장치 동작 방식**

1. **Fake Serial Device**

   * 실제 Serial 포트가 없으면 가상 장치로 동작.
   * 녹화 명령(`0x02,0x00,0x02,0x00,0x03`)을 받으면 **10초 동안 랜덤 오디오 데이터**를 생성.

     * 형식: `(frames, 32 channels)` 랜덤 값 (NumPy로 생성).
   * `ser.read(1926)` 호출 시 가짜 데이터를 동일한 크기로 반환.

2. **Fake Webcam**

   * 실제 USB 웹캠이 없으면 **OpenCV로 색상 패턴이 변하는 가상 영상**을 생성.

     * 예: 움직이는 그라디언트 배경이나 테스트 패턴 영상.
   * 녹화 중에는 30fps로 가짜 프레임을 생성하여 MP4로 저장.

3. **장치 감지**

   * 프로그램 시작 시:

     * **PySerial**로 포트 스캔 (`serial.tools.list_ports.comports()`).
     * **OpenCV**로 USB 카메라 연결 시도 (`cv2.VideoCapture(0)`).
   * 둘 중 하나라도 없으면 **자동으로 Fake 모드로 전환**하고 GUI에 표시.

---

### **3. 주요 모듈 구성 (수정 포함)**

#### **3.1. SerialManager**

* 실제 장치 연결 시 PySerial 사용.
* 없을 경우 `FakeSerialDevice` 사용:

  * 녹화 명령 시 랜덤 데이터 생성.
  * `read()` 호출 시 가짜 데이터 반환.

#### **3.2. CameraRecorder**

* 실제 USB 카메라 연결 시 OpenCV 사용.
* 없을 경우 `FakeCameraDevice` 사용:

  * 30fps로 움직이는 색상 패턴 생성.
  * MP4 파일로 정상 저장.

#### **3.3. Record Tab**

* **장치 상태 표시** (Real/Fake 여부 명확히 보여줌).
* Fake 모드일 경우에도 **녹화, 저장, 실시간 FFT 그래프** 동작.

---

### **4. Analysis Tab (변경 없음)**

* 저장된 영상/오디오 파일을 불러 Heatmap Overlay 분석.
* Heatmap 및 Colorbar의 Alpha 조절.
* 영상만 또는 Heatmap만 확인 가능.

---

### **5. Audio Tab (변경 없음)**

* WAV 파일 로드, 채널별 STFT/FFT/Waveform 분석.
* 멀티채널/단일채널 선택 가능.

---

### **6. 동작 시나리오 (Fake 지원 포함)**

1. 프로그램 실행 시:

   * Serial 포트와 USB 카메라 자동 감지.
   * 없으면 각각 FakeSerialDevice, FakeCameraDevice로 대체.
   * Record Tab에 **\[Fake Mode] 표시**.
2. 저장 폴더 선택 후 "Start Recording" 클릭.
3. 녹화:

   * 실제 장치 있으면 실제 데이터 수집.
   * 없으면 랜덤 오디오 데이터 + 테스트 영상 생성.
4. 녹화 종료 후 WAV/MP4 파일 저장.
5. Analysis Tab / Audio Tab 기능 동일하게 작동.

---

### **7. 구현 시 추가 고려사항**

* Fake 모드 여부는 **각 모듈 내에서 자동 처리** (메인 로직 수정 불필요).
* 저장되는 WAV/MP4 파일은 Fake 모드에서도 동일한 형식으로 저장.
* UI에서 **장치 상태 표시 (예: "Serial: Fake", "Camera: Real")**.

---

### **8. 추가 클래스**

* `FakeSerialDevice`:

  * `write(command)` 시 10초 녹화 명령만 해석, 랜덤 데이터 생성.
  * `read(n)` 호출 시 가짜 데이터 반환.
* `FakeCameraDevice`:

  * 30fps로 색상 변화 영상 생성 (테스트 패턴).
  * OpenCV VideoWriter와 동일한 API 제공.

---

원하면,

1. \*\*이 명세서 기반으로 프로젝트 초기 코드 (Record Tab + Fake 지원)\*\*부터 줄까?
2. \*\*Analysis Tab (Heatmap Overlay 포함)\*\*까지 통합한 코드로 바로 줄까?
3. 아니면 \*\*Record + Analysis + Audio Tab 풀버전 초기 코드 (Fake 포함)\*\*으로 한 번에 만들어줄까?

어느 방식으로 갈래?
