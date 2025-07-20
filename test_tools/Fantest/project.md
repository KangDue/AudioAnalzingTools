아래는 \*\*Windows 기반 노트북에서 BIOS/Embedded Controller(EC)를 통해 Fan 정보를 읽고, Level(1\~5)로 속도를 제어하거나 원래 Auto 모드로 복귀시키는 범용 C++ 프로그램을 작성하기 위한 개발 지시서 (Instruction)\*\*입니다.

> **주의:**
>
> * Windows 표준 API에는 BIOS나 EC의 Fan 제어 기능이 없으므로, \*\*ACPI EC I/O 또는 WMI (OEM API)\*\*를 활용해야 함.
> * 노트북 제조사마다 Fan 테이블 접근 방식이 다르므로, **범용 프로그램은 ACPI/WMI 기반 추상화 레이어를 사용**해야 함.
> * Intel, AMD, Qualcomm 모두 **ACPI Embedded Controller (EC) 인터페이스**를 통해 접근 가능.
> * 레벨별 Fan 속도는 보통 **BIOS에서 정의된 PWM Duty Table**을 읽어와 매핑해야 함.

---

## **개발 지시서 (Instruction)**

### **1. 주요 목표**

1. BIOS/EC로부터 Fan 지원 정보(Fan 개수, 제어 가능 여부, Level 수, Auto Mode 지원 여부) 읽기.
2. Fan Speed Levels (1\~5)에 따라 PWM Duty (혹은 RPM) 설정.
3. Auto 모드로 전환 (BIOS 기본 제어로 복귀).
4. 모든 동작을 C++로 구현, Intel/AMD/Qualcomm Windows 노트북에서 공통적으로 동작하도록 ACPI/WMI 사용.

---

### **2. 접근 방식**

1. **ACPI Embedded Controller (EC) 접근**

   * Windows에서 `DeviceIoControl`을 사용해 `\\.\ACPIEC` 또는 `\\.\PhysicalDrive0` 같은 경로로 IOCTL 호출.
   * EC 메모리 맵에서 Fan 관련 레지스터 (보통 `0x50~0x5F`) 읽고 쓰기.
   * Fan Level 테이블 (1\~5 레벨)과 Auto Mode 플래그 확인.

2. **WMI (Windows Management Instrumentation) 경로 (OEM 제공)**

   * `root\WMI` 네임스페이스에서 `ACPI\FanInformation`, `ACPI\ThermalZone` 클래스 사용.
   * Lenovo, Dell, HP는 `WmiMonitorBrightnessMethods` 유사한 **FanControl 클래스**를 제공.
   * WMI 호출로 Auto/Manual 모드 및 Fan Speed Level 설정.

---

### **3. 프로그램 구조**

1. **FanController 클래스**

   * 초기화 시 BIOS/EC에서 Fan 정보 테이블 읽기.
   * `GetFanInfo()` : Fan 개수, 지원 Level, 현재 속도/RPM, Auto Mode 여부 반환.
   * `SetFanLevel(fanId, level)` : Level(1\~5)로 PWM/RPM 설정.
   * `SetAutoMode(fanId)` : Auto 모드로 복귀.

2. **WMI + EC Wrapper**

   * 우선 WMI 인터페이스로 제어 시도, 실패 시 EC 직접 접근.
   * EC 접근 시 `CreateFile("\\\\.\\ACPIEC", ...)` + `DeviceIoControl`로 I/O 수행.

3. **CLI 인터페이스 (main)**

   * `fanctl info` → Fan 상태 출력.
   * `fanctl set <fanId> <level>` → Fan 레벨 제어.
   * `fanctl auto <fanId>` → Auto 모드로 전환.

---

### **4. C++ 코드 스켈레톤**

```cpp
#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <comdef.h>
#include <wbemidl.h>

#pragma comment(lib, "wbemuuid.lib")

struct FanInfo {
    int id;
    int currentRpm;
    int currentLevel;
    bool autoMode;
    int maxLevel;
};

class FanController {
public:
    FanController() { initWmi(); }
    ~FanController() { cleanupWmi(); }

    std::vector<FanInfo> GetFanInfo();
    bool SetFanLevel(int fanId, int level);  // 1~5
    bool SetAutoMode(int fanId);

private:
    bool initWmi();
    void cleanupWmi();
    bool queryFanData(std::vector<FanInfo>& fans);
    bool callFanMethod(int fanId, int level, bool autoMode);

    IWbemServices* pSvc = nullptr;
    IWbemLocator* pLoc = nullptr;
};

bool FanController::initWmi() {
    HRESULT hr = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hr)) return false;
    hr = CoInitializeSecurity(NULL, -1, NULL, NULL,
                               RPC_C_AUTHN_LEVEL_DEFAULT,
                               RPC_C_IMP_LEVEL_IMPERSONATE,
                               NULL, EOAC_NONE, NULL);
    if (FAILED(hr)) return false;
    hr = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER,
                          IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hr)) return false;
    hr = pLoc->ConnectServer(_bstr_t(L"ROOT\\WMI"), NULL, NULL, 0, NULL, 0, 0, &pSvc);
    return SUCCEEDED(hr);
}

void FanController::cleanupWmi() {
    if (pSvc) pSvc->Release();
    if (pLoc) pLoc->Release();
    CoUninitialize();
}

std::vector<FanInfo> FanController::GetFanInfo() {
    std::vector<FanInfo> fans;
    queryFanData(fans);
    return fans;
}

bool FanController::SetFanLevel(int fanId, int level) {
    return callFanMethod(fanId, level, false);
}

bool FanController::SetAutoMode(int fanId) {
    return callFanMethod(fanId, 0, true);
}

bool FanController::queryFanData(std::vector<FanInfo>& fans) {
    // OEM별 FanControl 클래스 (예: Lenovo "LENOVO_FanDevice", Dell "DELL_FanDevice") 쿼리
    IEnumWbemClassObject* pEnumerator = nullptr;
    HRESULT hr = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t("SELECT * FROM FanControlInformation"),  // OEM 문서에 따라 수정
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        NULL, &pEnumerator);
    if (FAILED(hr)) return false;

    IWbemClassObject* pObj = nullptr;
    ULONG returned = 0;
    while (pEnumerator && SUCCEEDED(pEnumerator->Next(WBEM_INFINITE, 1, &pObj, &returned)) && returned) {
        VARIANT vtProp;
        FanInfo info{};
        pObj->Get(L"FanID", 0, &vtProp, 0, 0);
        info.id = vtProp.intVal; VariantClear(&vtProp);
        pObj->Get(L"CurrentRPM", 0, &vtProp, 0, 0);
        info.currentRpm = vtProp.intVal; VariantClear(&vtProp);
        pObj->Get(L"CurrentLevel", 0, &vtProp, 0, 0);
        info.currentLevel = vtProp.intVal; VariantClear(&vtProp);
        pObj->Get(L"AutoMode", 0, &vtProp, 0, 0);
        info.autoMode = vtProp.boolVal; VariantClear(&vtProp);
        info.maxLevel = 5;
        fans.push_back(info);
        pObj->Release();
    }
    if (pEnumerator) pEnumerator->Release();
    return !fans.empty();
}

bool FanController::callFanMethod(int fanId, int level, bool autoMode) {
    // OEM WMI 메서드 호출 (예: SetFanSpeed)
    IEnumWbemClassObject* pEnumerator = nullptr;
    HRESULT hr = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t("SELECT * FROM FanControlMethods"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        NULL, &pEnumerator);
    if (FAILED(hr)) return false;

    IWbemClassObject* pObj = nullptr;
    ULONG returned = 0;
    if (pEnumerator && SUCCEEDED(pEnumerator->Next(WBEM_INFINITE, 1, &pObj, &returned)) && returned) {
        VARIANT varId, varLevel, varAuto;
        varId.vt = VT_I4; varId.intVal = fanId;
        varLevel.vt = VT_I4; varLevel.intVal = level;
        varAuto.vt = VT_BOOL; varAuto.boolVal = autoMode ? VARIANT_TRUE : VARIANT_FALSE;

        // 실제 호출 부분은 OEM 문서 참고 필요
        // pObj->ExecMethod(L"SetFanSpeed", ...);

        pObj->Release();
    }
    if (pEnumerator) pEnumerator->Release();
    return true;
}

int main(int argc, char** argv) {
    FanController ctl;
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  fanctl info\n"
                  << "  fanctl set <fanId> <level (1-5)>\n"
                  << "  fanctl auto <fanId>\n";
        return 0;
    }
    std::string cmd = argv[1];
    if (cmd == "info") {
        auto fans = ctl.GetFanInfo();
        for (auto& f : fans) {
            std::cout << "Fan " << f.id << ": "
                      << (f.autoMode ? "[AUTO]" : "[MANUAL]")
                      << " Level=" << f.currentLevel
                      << " RPM=" << f.currentRpm << "\n";
        }
    } else if (cmd == "set" && argc >= 4) {
        int id = atoi(argv[2]);
        int level = atoi(argv[3]);
        ctl.SetFanLevel(id, level);
    } else if (cmd == "auto" && argc >= 3) {
        int id = atoi(argv[2]);
        ctl.SetAutoMode(id);
    }
    return 0;
}
```

---

### **이 지시서로 구현할 때 주의할 점**

1. **FanControlInformation, FanControlMethods** 클래스 이름은 OEM마다 다르므로,
   실제 장치의 WMI 네임스페이스를 `wbemtest.exe`로 확인 후 수정 필요.
2. WMI로 제어가 불가능하면, **ACPI EC 메모리 직접 접근**을 구현해야 함.

   * 이 경우, `\\.\ACPIEC`를 열어 `DeviceIoControl`로 레지스터 읽고 쓰는 코드 추가 필요.
3. 권한: 관리자 권한으로 실행 필요.
4. 범용성을 위해 **WMI → EC 순서로 시도**하는 구조를 권장.