# 가속도계 + 오디오 동시 녹음 시스템

이 프로젝트는 Windows 환경에서 가속도계 센서와 마이크를 동시에 사용하여 데이터를 수집하고 저장하는 C++ 애플리케이션입니다.

## 📋 기능

- **가속도계 데이터 수집**: Windows Sensor API를 사용하여 3축 가속도 데이터를 CSV 형식으로 저장
- **오디오 녹음**: WASAPI를 사용하여 모든 활성 마이크에서 동시 녹음, WAV 형식으로 저장
- **동시 실행**: 가속도계와 오디오 녹음이 정확히 동시에 시작되고 종료됨
- **오디오 이펙트 비활성화**: Windows의 자동 노이즈 억제, 에코 제거 등의 효과 비활성화
- **타임스탬프 기반 파일명**: 녹음 시작 시간을 기반으로 한 고유 파일명 생성

## 🛠️ 시스템 요구사항

### 운영체제
- **Windows 10 이상** (Windows 11 권장)
- 64비트 시스템

### 하드웨어
- **가속도계 센서**: HID over I²C 센서 (장치 관리자에서 "Sensors"로 표시)
- **마이크**: 내장 또는 외장 마이크 (여러 개 지원)

### 개발 환경
- **Visual Studio 2019 이상** 또는 **Visual Studio 2022** (Community/Professional/Enterprise)
- **CMake 3.16 이상**
- **Windows SDK 10.0.19041.0 이상**

## 📦 프로젝트 구조

```
acc_reader/
├── main.cpp              # 메인 프로그램 (동시 실행 제어)
├── acc.cpp               # 가속도계 데이터 수집 구현
├── acc.h                 # 가속도계 클래스 헤더
├── audio.cpp             # 오디오 녹음 구현
├── audio.h               # 오디오 녹음 클래스 헤더
├── CMakeLists.txt        # CMake 빌드 설정
├── README.md             # 프로젝트 문서 (이 파일)
└── work_log.txt          # 개발 작업 일지
```

## 🚀 빌드 및 실행 가이드

### 1단계: 개발 환경 설정

#### Visual Studio 설치
1. [Visual Studio 다운로드 페이지](https://visualstudio.microsoft.com/downloads/)에서 Visual Studio 2022 Community 다운로드
2. 설치 시 다음 워크로드 선택:
   - **C++를 사용한 데스크톱 개발**
   - **CMake용 Visual C++ 도구**

#### CMake 설치
1. [CMake 다운로드 페이지](https://cmake.org/download/)에서 Windows용 설치 프로그램 다운로드
2. 설치 시 "Add CMake to the system PATH" 옵션 선택

### 2단계: 프로젝트 다운로드 및 설정

```bash
# 프로젝트 폴더로 이동
cd C:\path\to\your\project\folder

# 또는 Git을 사용하는 경우
git clone <repository-url>
cd acc_reader
```

### 3단계: CMake를 사용한 빌드

#### 명령줄에서 빌드
```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# CMake 구성 (Release 모드)
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

# 빌드 실행
cmake --build . --config Release
```

#### Visual Studio에서 빌드
1. Visual Studio 실행
2. **파일 > 열기 > CMake...** 선택
3. 프로젝트 폴더의 `CMakeLists.txt` 파일 선택
4. Visual Studio가 자동으로 CMake 구성을 생성
5. **빌드 > 모두 빌드** 또는 `Ctrl+Shift+B`

### 4단계: 실행

```bash
# 빌드된 실행 파일 위치로 이동
cd build\bin

# 기본 실행 (10초 녹음)
.\AccelerometerAudioRecorder.exe

# 사용자 정의 녹음 시간 (예: 30초)
.\AccelerometerAudioRecorder.exe 30
```

## 📊 출력 파일 형식

### 가속도계 데이터 (CSV)
파일명: `accel_YYYYMMDD_HHMMSS_mmm.csv`

```csv
time_ms,X(g),Y(g),Z(g)
0,0.012,-0.987,0.156
20,0.015,-0.985,0.158
40,0.011,-0.989,0.154
...
```

- **time_ms**: 녹음 시작부터의 경과 시간 (밀리초)
- **X(g), Y(g), Z(g)**: 각 축의 가속도 (중력가속도 단위, 1g ≈ 9.81 m/s²)

### 오디오 데이터 (WAV)
파일명: `audio_YYYYMMDD_HHMMSS_mmm_mic0.wav`, `audio_YYYYMMDD_HHMMSS_mmm_mic1.wav`, ...

- **포맷**: PCM, 16비트, 44.1kHz, 모노
- **각 마이크별로 별도 파일 생성**
- **오디오 이펙트 비활성화됨** (원본 음성 데이터)

## ⚙️ 설정 옵션

### 가속도계 설정
`main.cpp`에서 수정 가능:
```cpp
accelerometer.SetSampleRate(50);  // 샘플링 주파수 (Hz)
```

### 오디오 설정
`main.cpp`에서 수정 가능:
```cpp
audioRecorder.SetAudioFormat(44100, 1, 16);  // 샘플레이트, 채널수, 비트깊이
```

## 🔧 문제 해결

### 가속도계 관련 문제

**"No accelerometer found" 오류**
1. 장치 관리자에서 "센서" 카테고리 확인
2. 가속도계가 활성화되어 있는지 확인
3. Windows 업데이트로 센서 드라이버 최신화

**값이 최대/최소로만 나오는 경우**
- 센서 드라이버 재설치
- 다른 센서 애플리케이션 종료

### 오디오 관련 문제

**"No microphones found" 오류**
1. 마이크가 연결되어 있는지 확인
2. Windows 설정 > 개인정보 > 마이크에서 앱 접근 허용
3. 장치 관리자에서 오디오 드라이버 확인

**오디오 파일이 생성되지 않는 경우**
- 마이크 권한 확인
- 다른 오디오 애플리케이션 종료
- Windows 오디오 서비스 재시작

### 빌드 관련 문제

**CMake 오류**
- CMake 버전 확인 (3.16 이상 필요)
- Visual Studio Build Tools 설치 확인

**링크 오류**
- Windows SDK 설치 확인
- 프로젝트를 관리자 권한으로 실행

## 🔒 권한 요구사항

이 애플리케이션은 다음 권한이 필요합니다:
- **센서 접근 권한**: 가속도계 데이터 읽기
- **마이크 접근 권한**: 오디오 녹음
- **파일 시스템 쓰기 권한**: 데이터 파일 저장

Windows 10/11에서는 처음 실행 시 권한 요청 대화상자가 나타날 수 있습니다.

## 📈 성능 최적화

### 권장 설정
- **가속도계 샘플링**: 50Hz (일반적인 움직임 분석에 충분)
- **오디오 샘플링**: 44.1kHz (CD 품질)
- **녹음 시간**: 60초 이하 (메모리 사용량 고려)

### 시스템 리소스
- **RAM**: 최소 4GB, 권장 8GB 이상
- **저장공간**: 1분 녹음당 약 5-10MB
- **CPU**: 멀티코어 프로세서 권장 (동시 처리)

## 🤝 기여 및 개발

### 코드 수정 가이드
1. `acc.cpp/acc.h`: 가속도계 관련 기능 수정
2. `audio.cpp/audio.h`: 오디오 녹음 관련 기능 수정
3. `main.cpp`: 전체 프로그램 흐름 및 UI 수정

### 새로운 센서 추가
1. 새로운 센서 클래스 생성 (예: `gyro.cpp`, `gyro.h`)
2. `main.cpp`에서 동시 실행 로직 추가
3. `CMakeLists.txt`에 새 소스 파일 추가

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 상업적 사용 시 관련 라이브러리의 라이선스를 확인하시기 바랍니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 이 README의 문제 해결 섹션 확인
2. Windows 이벤트 로그 확인
3. 프로젝트 이슈 트래커에 문제 보고

---

**마지막 업데이트**: 2024년 12월
**버전**: 1.0.0