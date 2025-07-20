알겠어.
**노트북의 모든 마이크(내장, 외장 포함)** 를 동시에 열어서 **WAV로 저장**하면서,
Windows가 자동으로 거는 **Noise Suppression, Echo Cancellation, Voice Processing 같은 오디오 이펙트**를 꺼버리는 코드(C++)를 만들게.

이를 위해서는:

* **WASAPI (Windows Audio Session API)** 를 이용해서 모든 입력 디바이스 열기
* `IAudioClient`로 각 디바이스 스트림 초기화
* `IAudioEffectsManager` 또는 `IAudioClientProperties`를 이용해서 **시스템 오디오 이펙트 비활성화**
* 각 디바이스에서 PCM 스트림을 읽어 **WAV로 각각 저장**

---

## **중요 포인트**

1. **Windows 기본 마이크는 자동으로 DSP (Noise Suppression, AGC 등)** 가 활성화돼 있음.

   * 이를 끄려면 `IAudioClient::SetClientProperties`에서 `AudioCategory_Other`를 사용하거나,
     **AudioEffects**를 수동으로 비활성화해야 함.
2. **모든 마이크 열기**

   * `IMMDeviceEnumerator::EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, …)`로 전부 나열.
3. **동시에 녹음**

   * 각 입력 장치별로 별도의 `IAudioClient`를 생성하고, 멀티스레드로 데이터 캡처.
4. **WAV 저장**

   * 표준 PCM 16bit/44.1kHz로 변환해서 `.wav` 파일로 저장.

---

## **C++ 예제 (모든 마이크 녹음 + WAV 저장, 이펙트 비활성화)**

> 이 코드는 \*\*각 마이크를 별도의 WAV 파일(`mic0.wav`, `mic1.wav` …)\*\*로 저장.
> WASAPI Exclusive 모드가 아닌 **Shared 모드**를 사용해 호환성을 높임.

```cpp
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <functiondiscoverykeys_devpkey.h>
#include <propvarutil.h>
#include <avrt.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

constexpr int RECORD_SECONDS = 5;  // 녹음 시간
constexpr int SAMPLE_RATE = 44100; // 44.1kHz
constexpr int CHANNELS = 1;        // 모노 저장
constexpr int BITS_PER_SAMPLE = 16;

// WAV 파일 헤더 작성
void WriteWavHeader(std::ofstream& file, int totalSamples) {
    int byteRate = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE / 8);
    int blockAlign = CHANNELS * (BITS_PER_SAMPLE / 8);
    int dataChunkSize = totalSamples * CHANNELS * (BITS_PER_SAMPLE / 8);
    int riffChunkSize = 36 + dataChunkSize;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&riffChunkSize), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);

    int subchunk1Size = 16;
    short audioFormat = 1;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&CHANNELS), 2);
    file.write(reinterpret_cast<const char*>(&SAMPLE_RATE), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&BITS_PER_SAMPLE), 2);

    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataChunkSize), 4);
}

// 특정 마이크를 녹음
void RecordMic(IMMDevice* pDevice, int index, std::atomic<bool>& running) {
    IAudioClient* pAudioClient = nullptr;
    IAudioCaptureClient* pCaptureClient = nullptr;
    WAVEFORMATEX* pwfx = nullptr;

    HRESULT hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    if (FAILED(hr)) return;

    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) return;

    // 녹음 포맷 강제 변환 (모노, 16bit, 44.1kHz)
    pwfx->wFormatTag = WAVE_FORMAT_PCM;
    pwfx->nChannels = CHANNELS;
    pwfx->nSamplesPerSec = SAMPLE_RATE;
    pwfx->wBitsPerSample = BITS_PER_SAMPLE;
    pwfx->nBlockAlign = (pwfx->nChannels * pwfx->wBitsPerSample) / 8;
    pwfx->nAvgBytesPerSec = pwfx->nSamplesPerSec * pwfx->nBlockAlign;
    pwfx->cbSize = 0;

    // 오디오 이펙트(Noise Suppression 등) 비활성화
    AUDIOCLIENT_PROPERTIES props = {};
    props.cbSize = sizeof(props);
    props.bIsOffload = FALSE;
    props.eCategory = AudioCategory_Other; // 통화용 아님, 이펙트 해제
    props.Options = AUDCLNT_STREAMOPTIONS_RAW; // RAW 모드
    pAudioClient->SetClientProperties(&props);

    REFERENCE_TIME hnsRequestedDuration = 10000000; // 1초 버퍼
    hr = pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                  AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                  hnsRequestedDuration, 0, pwfx, NULL);
    if (FAILED(hr)) return;

    hr = pAudioClient->GetService(IID_PPV_ARGS(&pCaptureClient));
    if (FAILED(hr)) return;

    pAudioClient->Start();

    std::vector<BYTE> buffer;
    std::ofstream wavFile("mic" + std::to_string(index) + ".wav", std::ios::binary);
    int totalSamples = SAMPLE_RATE * RECORD_SECONDS;
    buffer.reserve(totalSamples * pwfx->nBlockAlign);

    auto start = std::chrono::steady_clock::now();
    while (running) {
        UINT32 packetLength = 0;
        pCaptureClient->GetNextPacketSize(&packetLength);
        while (packetLength) {
            BYTE* pData;
            UINT32 numFrames;
            DWORD flags;
            pCaptureClient->GetBuffer(&pData, &numFrames, &flags, NULL, NULL);

            size_t bytesToCopy = numFrames * pwfx->nBlockAlign;
            buffer.insert(buffer.end(), pData, pData + bytesToCopy);

            pCaptureClient->ReleaseBuffer(numFrames);
            pCaptureClient->GetNextPacketSize(&packetLength);
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= RECORD_SECONDS) {
            running = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // WAV 파일 저장
    WriteWavHeader(wavFile, totalSamples);
    wavFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    wavFile.close();

    pAudioClient->Stop();
    CoTaskMemFree(pwfx);
    pCaptureClient->Release();
    pAudioClient->Release();
}

int main() {
    CoInitialize(NULL);

    IMMDeviceEnumerator* pEnumerator = nullptr;
    IMMDeviceCollection* pDevices = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL,
                                  IID_PPV_ARGS(&pEnumerator));
    if (FAILED(hr)) return -1;

    hr = pEnumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &pDevices);
    if (FAILED(hr)) return -1;

    UINT count;
    pDevices->GetCount(&count);
    if (count == 0) {
        std::cout << "No microphones found.\n";
        return 0;
    }

    std::vector<std::thread> threads;
    std::atomic<bool> running(true);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* pDevice = nullptr;
        pDevices->Item(i, &pDevice);
        threads.emplace_back(RecordMic, pDevice, i, std::ref(running));
    }

    for (auto& t : threads) t.join();

    pDevices->Release();
    pEnumerator->Release();
    CoUninitialize();
    return 0;
}
```

---

### **이 코드가 하는 일**

1. **현재 활성화된 모든 마이크 열기** (`IMMDeviceEnumerator`)
2. 각 마이크마다 **WASAPI 캡처** 시작 (멀티스레드)
3. **Windows 오디오 이펙트 (Noise Suppression, AGC, Echo Cancellation)** 비활성화

   * `SetClientProperties` + `AUDCLNT_STREAMOPTIONS_RAW`
4. **5초간 PCM 데이터를 캡처해서 `mic0.wav`, `mic1.wav` …로 저장**

---

### **확장 가능**

* 모든 마이크를 **동일한 WAV 파일에 믹싱**해서 저장하도록 변경 가능
* 녹음 시간을 명령행 인자로 조절 가능하게 만들 수 있음
* 샘플레이트, 채널수, 비트 깊이를 옵션화 가능

---

**바로 실행 가능한 Visual Studio 프로젝트 세팅 가이드**도 줄까?
아니면 **모든 마이크를 하나로 믹싱해서 WAV 저장하는 버전**으로 바꿔줄까?
혹은 **녹음 시간, 샘플레이트를 CLI 인자로 넣는 버전**으로? 어느 쪽?
