# Fan Control Utility

범용 Windows 노트북 팬 속도 제어 프로그램입니다. Intel, AMD, Qualcomm 기반 노트북에서 BIOS/EC를 통해 팬 속도를 제어할 수 있습니다.

## 주요 기능

- **팬 정보 조회**: 현재 팬 상태, RPM, 레벨 확인
- **팬 속도 제어**: 1~5 레벨로 팬 속도 수동 설정
- **자동 모드**: BIOS 기본 제어로 복귀
- **범용 지원**: WMI → EC 순서로 시도하여 다양한 노트북 지원
- **관리자 권한 자동 요청**: UAC를 통한 권한 상승

## 시스템 요구사항

- **운영체제**: Windows 10/11 (64비트 권장)
- **권한**: 관리자 권한 필요
- **개발환경**: Visual Studio 2022
- **플랫폼**: x86, x64 지원

## Visual Studio 2022 빌드 가이드

### 1. 프로젝트 열기

1. Visual Studio 2022를 실행합니다.
2. `FanControl.sln` 파일을 엽니다.
3. 또는 "파일" → "열기" → "프로젝트/솔루션"에서 `FanControl.sln`을 선택합니다.

### 2. 빌드 구성 설정

#### Debug 빌드 (개발/테스트용)
```
1. 상단 툴바에서 "Debug" 선택
2. 플랫폼을 "x64" 또는 "x86" 선택
3. "빌드" → "솔루션 빌드" (Ctrl+Shift+B)
```

#### Release 빌드 (배포용)
```
1. 상단 툴바에서 "Release" 선택
2. 플랫폼을 "x64" 선택 (권장)
3. "빌드" → "솔루션 빌드" (Ctrl+Shift+B)
```

### 3. 빌드 출력 위치

빌드가 완료되면 다음 위치에 실행 파일이 생성됩니다:

- **Debug 빌드**: `Debug/fanctl.exe`
- **Release 빌드**: `Release/fanctl.exe` 또는 `x64/Release/fanctl.exe`

### 4. 실행 및 테스트

1. **관리자 권한으로 실행**:
   - 생성된 `fanctl.exe`를 우클릭
   - "관리자 권한으로 실행" 선택

2. **기본 테스트**:
   ```cmd
   fanctl info
   ```

## 사용법

### 기본 명령어

```cmd
# 팬 정보 확인
fanctl info

# 팬 속도 설정 (팬 ID 1, 레벨 3)
fanctl set 1 3

# 자동 모드로 전환 (팬 ID 1)
fanctl auto 1

# 도움말 표시
fanctl help
```

### 사용 예시

```cmd
C:\> fanctl info

=== Fan Information ===
Fan ID Mode     Level   RPM     Max Level   Status
------------------------------------------------------------
1      AUTO     2       2400    5           EC Direct Access
2      MANUAL   4       3200    5           EC Direct Access

C:\> fanctl set 1 5
Setting fan 1 to level 5...
Success: Fan 1 set to level 5.
Current status: Level 5, RPM 4200, Mode MANUAL

C:\> fanctl auto 1
Setting fan 1 to auto mode...
Success: Fan 1 set to auto mode.
Current status: Level 2, RPM 2400, Mode AUTO
```

## 릴리스 배포 가이드

### 1. Release 빌드 생성

1. Visual Studio에서 "Release" + "x64" 구성으로 빌드
2. `x64/Release/fanctl.exe` 파일 확인

### 2. 배포 패키지 준비

다음 파일들을 포함하여 배포 패키지를 만듭니다:

```
FanControl_v1.0/
├── fanctl.exe          # 메인 실행 파일
├── README.md           # 사용 설명서
├── LICENSE.txt         # 라이선스 (선택사항)
└── install.bat         # 설치 스크립트 (선택사항)
```

### 3. 설치 스크립트 (install.bat)

```batch
@echo off
echo Fan Control Utility Installer
echo.

:: 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrator privileges confirmed.
) else (
    echo This installer requires administrator privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

:: 프로그램 복사
echo Installing fanctl.exe to C:\Windows\System32\
copy fanctl.exe C:\Windows\System32\

if %errorLevel% == 0 (
    echo Installation completed successfully!
    echo You can now use 'fanctl' command from anywhere.
) else (
    echo Installation failed.
)

echo.
echo Testing installation...
fanctl help

pause
```

### 4. 디지털 서명 (선택사항)

배포용 실행 파일에 디지털 서명을 추가하려면:

1. 코드 서명 인증서 획득
2. Visual Studio의 "프로젝트 속성" → "링커" → "고급"에서 서명 설정
3. 또는 `signtool.exe`를 사용하여 빌드 후 서명

## 문제 해결

### 빌드 오류

1. **"wbemuuid.lib를 찾을 수 없음"**:
   - Windows SDK가 설치되어 있는지 확인
   - Visual Studio Installer에서 "Windows 10/11 SDK" 구성 요소 설치

2. **"관리자 권한 필요" 오류**:
   - Visual Studio를 관리자 권한으로 실행
   - 또는 UAC 설정을 프로젝트 속성에서 확인

### 실행 오류

1. **"팬 컨트롤러 초기화 실패"**:
   - 관리자 권한으로 실행했는지 확인
   - 노트북이 팬 제어를 지원하는지 확인
   - BIOS/UEFI 업데이트 고려

2. **"팬이 감지되지 않음"**:
   - 노트북 제조사별 드라이버 설치 확인
   - WMI 서비스가 실행 중인지 확인
   - EC 접근이 차단되어 있는지 확인

## 지원되는 노트북

### 테스트된 브랜드
- Lenovo ThinkPad 시리즈
- Dell Latitude/XPS 시리즈
- HP EliteBook/ProBook 시리즈
- ASUS 비즈니스 노트북
- MSI 게이밍 노트북

### 지원되는 프로세서
- Intel Core 시리즈 (6세대 이상)
- AMD Ryzen 시리즈
- Qualcomm Snapdragon (Windows on ARM)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

1. 이슈 리포트: 버그나 개선사항을 GitHub Issues에 등록
2. 풀 리퀘스트: 코드 개선이나 새로운 기능 추가
3. 테스트: 다양한 노트북 모델에서 테스트 결과 공유

## 주의사항

⚠️ **경고**: 이 프로그램은 하드웨어를 직접 제어합니다. 잘못된 사용으로 인한 하드웨어 손상에 대해서는 책임지지 않습니다.

- 팬 속도를 너무 낮게 설정하면 과열이 발생할 수 있습니다.
- 시스템이 불안정해지면 즉시 자동 모드로 전환하세요.
- 중요한 작업 중에는 자동 모드 사용을 권장합니다.

## 연락처

문의사항이나 지원이 필요한 경우:
- GitHub Issues: 버그 리포트 및 기능 요청
- Email: [개발자 이메일]
- 문서: 이 README.md 파일 참조