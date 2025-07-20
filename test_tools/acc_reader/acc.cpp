#define _WIN32_DCOM
#include "acc.h"
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

Accelerometer::Accelerometer() : 
    pSensorManager(nullptr), 
    pSensors(nullptr), 
    pSensor(nullptr),
    isRecording(false),
    sampleRateHz(50) {
}

Accelerometer::~Accelerometer() {
    Stop();
    Cleanup();
}

bool Accelerometer::Initialize() {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed for accelerometer\n";
        return false;
    }

    hr = CoCreateInstance(CLSID_SensorManager, NULL, CLSCTX_INPROC_SERVER,
                          IID_PPV_ARGS(&pSensorManager));
    if (FAILED(hr)) {
        std::cerr << "Failed to create SensorManager\n";
        return false;
    }

    hr = pSensorManager->GetSensorsByType(SENSOR_TYPE_ACCELEROMETER_3D, &pSensors);
    if (FAILED(hr) || !pSensors) {
        std::cerr << "No accelerometer found\n";
        return false;
    }

    ULONG count = 0;
    pSensors->GetCount(&count);
    if (count == 0) {
        std::cerr << "No accelerometer sensors found\n";
        return false;
    }

    pSensors->GetAt(0, &pSensor);  // 첫 번째 가속도계 선택
    
    std::cout << "Accelerometer initialized successfully\n";
    return true;
}

void Accelerometer::Start(const std::string& filename, int durationSec) {
    if (!pSensor || isRecording) {
        return;
    }

    isRecording = true;
    recordingThread = std::thread(&Accelerometer::RecordingLoop, this, filename, durationSec);
}

void Accelerometer::Stop() {
    isRecording = false;
    if (recordingThread.joinable()) {
        recordingThread.join();
    }
}

void Accelerometer::SetSampleRate(int hz) {
    sampleRateHz = hz;
}

bool Accelerometer::IsRecording() const {
    return isRecording;
}

void Accelerometer::RecordingLoop(const std::string& filename, int durationSec) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Failed to open accelerometer output file: " << filename << std::endl;
        isRecording = false;
        return;
    }

    fout << "time_ms,X(g),Y(g),Z(g)\n";
    
    auto start = std::chrono::steady_clock::now();
    std::cout << "Accelerometer recording started...\n";
    
    while (isRecording) {
        auto now = std::chrono::steady_clock::now();
        auto elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsedSec >= durationSec) {
            break;
        }

        ISensorDataReport* pReport = nullptr;
        HRESULT hr = pSensor->GetData(&pReport);
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

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / sampleRateHz));
    }

    fout.close();
    isRecording = false;
    std::cout << "Accelerometer recording completed and saved to: " << filename << std::endl;
}

void Accelerometer::Cleanup() {
    SafeRelease(pSensor);
    SafeRelease(pSensors);
    SafeRelease(pSensorManager);
    CoUninitialize();
}