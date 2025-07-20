알겠어. 그러면 \*\*HID over I²C 센서(장치 관리자에서 "Sensors"로 보이는 가속도계)\*\*에서,
\*\*Sensor API (ISensorManager)**를 통해 데이터를 n초간 읽어서 CSV로 저장하는 C++ 예제**를 만들어줄게.

이 버전은 **레지스터 직접 접근 없이**, Windows가 제공하는 표준 센서 인터페이스를 사용해서 값을 읽어와.
또, **값이 항상 최대/최소로 고정되는 문제**는 보통 `PROPVARIANT` 타입 처리 실수 때문에 생기니, 이걸 명확히 처리하도록 할 거야.

---

## **C++ 예제 (Sensor API, COM 기반)**

이 예제는:

* Windows 10 이상에서 동작.
* 가속도계(SENSOR\_TYPE\_ACCELEROMETER\_3D)에서 3축 데이터를 읽음.
* 1초에 50Hz로 샘플링 (원하면 변경 가능).
* n초 동안 데이터를 CSV(`accel_data.csv`)로 저장.

```cpp
#define _WIN32_DCOM
#include <windows.h>
#include <sensorsapi.h>
#include <sensors.h>
#include <initguid.h>
#include <propvarutil.h>
#include <comdef.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

#pragma comment(lib, "sensorsapi.lib")
#pragma comment(lib, "propsys.lib")

// n초 동안 측정
constexpr int DURATION_SEC = 5;
constexpr int SAMPLE_RATE_HZ = 50;

void SafeRelease(IUnknown* p) {
    if (p) p->Release();
}

float GetFloatFromPROPVARIANT(const PROPVARIANT& var) {
    if (var.vt == VT_R4) {
        return var.fltVal;  // float 값
    } else if (var.vt == VT_R8) {
        return static_cast<float>(var.dblVal);  // double을 float로 변환
    } else if (var.vt == VT_I4) {
        return static_cast<float>(var.lVal);   // int32를 float로 변환
    } else if (var.vt == VT_UI4) {
        return static_cast<float>(var.ulVal);
    } else {
        return 0.0f;  // 알 수 없는 타입이면 0 반환
    }
}

int main() {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed\n";
        return -1;
    }

    ISensorManager* pSensorManager = nullptr;
    hr = CoCreateInstance(CLSID_SensorManager, NULL, CLSCTX_INPROC_SERVER,
                          IID_PPV_ARGS(&pSensorManager));
    if (FAILED(hr)) {
        std::cerr << "Failed to create SensorManager\n";
        CoUninitialize();
        return -1;
    }

    ISensorCollection* pSensors = nullptr;
    hr = pSensorManager->GetSensorsByType(SENSOR_TYPE_ACCELEROMETER_3D, &pSensors);
    if (FAILED(hr) || !pSensors) {
        std::cerr << "No accelerometer found\n";
        SafeRelease(pSensorManager);
        CoUninitialize();
        return -1;
    }

    ULONG count = 0;
    pSensors->GetCount(&count);
    if (count == 0) {
        std::cerr << "No accelerometer sensors found\n";
        SafeRelease(pSensors);
        SafeRelease(pSensorManager);
        CoUninitialize();
        return -1;
    }

    ISensor* pSensor = nullptr;
    pSensors->GetAt(0, &pSensor);  // 첫 번째 가속도계 선택

    // CSV 파일 열기
    std::ofstream fout("accel_data.csv");
    fout << "time_ms,X(g),Y(g),Z(g)\n";

    auto start = std::chrono::steady_clock::now();
    while (true) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsedSec >= DURATION_SEC) break;

        ISensorDataReport* pReport = nullptr;
        hr = pSensor->GetData(&pReport);
        if (SUCCEEDED(hr) && pReport) {
            PROPVARIANT varX, varY, varZ;
            PropVariantInit(&varX);
            PropVariantInit(&varY);
            PropVariantInit(&varZ);

            // X, Y, Z 값 가져오기
            pReport->GetSensorValue(SENSOR_DATA_TYPE_ACCELERATION_X_G, &varX);
            pReport->GetSensorValue(SENSOR_DATA_TYPE_ACCELERATION_Y_G, &varY);
            pReport->GetSensorValue(SENSOR_DATA_TYPE_ACCELERATION_Z_G, &varZ);

            float x = GetFloatFromPROPVARIANT(varX);
            float y = GetFloatFromPROPVARIANT(varY);
            float z = GetFloatFromPROPVARIANT(varZ);

            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            fout << timestamp << "," << x << "," << y << "," << z << "\n";

            PropVariantClear(&varX);
            PropVariantClear(&varY);
            PropVariantClear(&varZ);
            SafeRelease(pReport);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / SAMPLE_RATE_HZ));
    }

    fout.close();

    SafeRelease(pSensor);
    SafeRelease(pSensors);
    SafeRelease(pSensorManager);
    CoUninitialize();
    return 0;
}
```

---

## **핵심 포인트**

1. **값이 최대/최소로만 나오는 문제**

   * `PROPVARIANT`의 `vt` (타입) 확인 없이 단순히 `fltVal`만 읽으면 0, 1, ±65535 같은 고정값만 나올 수 있음.
   * 위 코드에서는 `VT_R4`, `VT_R8`, `VT_I4`, `VT_UI4` 모두 처리.

2. **단위**

   * `SENSOR_DATA_TYPE_ACCELERATION_X_G` → **중력가속도 단위(G)** 로 제공됨. (1G ≈ 9.81 m/s²)
   * 필요하면 9.81을 곱해 m/s² 단위로 변환 가능.

3. **센서 선택**

   * `GetSensorsByType(SENSOR_TYPE_ACCELEROMETER_3D)`로 3축 가속도계만 선택.
   * 여러 센서가 있으면 `GetAt()`으로 원하는 인덱스 선택 가능.

4. **HID over I²C라서 레지스터 제어는 불가**

   * 이 방식은 Windows가 제공하는 가속도계 API만 사용.

---

원하면 이걸 **클래스로 감싼 버전**이나
**여러 센서 동시 로깅 (가속도+자이로)** 버전으로 확장할 수도 있어.

---

### 다음 중 어떤 게 필요해?

1. **이대로 실행 가능한 프로젝트 설정 가이드 (Visual Studio 세팅 포함)**
2. **자이로스코프, 자기장 센서 같이 여러 센서 동시 로깅 버전**
3. **데이터 단위 변환 (m/s²) 및 추가 필터링 적용 버전**

어느 쪽으로 이어서 만들어줄까?
