#include "audio.h"
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

AudioRecorder::AudioRecorder() : 
    pEnumerator(nullptr),
    pDevices(nullptr),
    isRecording(false),
    sampleRate(44100),
    channels(1),
    bitsPerSample(16) {
}

AudioRecorder::~AudioRecorder() {
    Stop();
    Cleanup();
}

bool AudioRecorder::Initialize() {
    HRESULT hr = CoInitialize(NULL);
    if (FAILED(hr)) {
        std::cerr << "COM initialization failed for audio\n";
        return false;
    }

    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_ALL,
                          IID_PPV_ARGS(&pEnumerator));
    if (FAILED(hr)) {
        std::cerr << "Failed to create MMDeviceEnumerator\n";
        return false;
    }

    hr = pEnumerator->EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE, &pDevices);
    if (FAILED(hr)) {
        std::cerr << "Failed to enumerate audio endpoints\n";
        return false;
    }

    UINT count;
    pDevices->GetCount(&count);
    if (count == 0) {
        std::cerr << "No microphones found\n";
        return false;
    }

    std::cout << "Audio recorder initialized with " << count << " microphone(s)\n";
    return true;
}

void AudioRecorder::Start(const std::string& baseFilename, int durationSec) {
    if (isRecording) {
        return;
    }

    isRecording = true;
    recordingThread = std::thread(&AudioRecorder::RecordingLoop, this, baseFilename, durationSec);
}

void AudioRecorder::Stop() {
    isRecording = false;
    if (recordingThread.joinable()) {
        recordingThread.join();
    }
}

void AudioRecorder::SetAudioFormat(int sampleRate, int channels, int bitsPerSample) {
    this->sampleRate = sampleRate;
    this->channels = channels;
    this->bitsPerSample = bitsPerSample;
}

bool AudioRecorder::IsRecording() const {
    return isRecording;
}

void AudioRecorder::WriteWavHeader(std::ofstream& file, int totalSamples) {
    int byteRate = sampleRate * channels * (bitsPerSample / 8);
    int blockAlign = channels * (bitsPerSample / 8);
    int dataChunkSize = totalSamples * channels * (bitsPerSample / 8);
    int riffChunkSize = 36 + dataChunkSize;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&riffChunkSize), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);

    int subchunk1Size = 16;
    short audioFormat = 1;
    file.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&channels), 2);
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataChunkSize), 4);
}

void AudioRecorder::RecordMic(IMMDevice* pDevice, int index, const std::string& filename, int durationSec) {
    IAudioClient* pAudioClient = nullptr;
    IAudioCaptureClient* pCaptureClient = nullptr;
    WAVEFORMATEX* pwfx = nullptr;

    HRESULT hr = pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, NULL, (void**)&pAudioClient);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client for mic " << index << std::endl;
        return;
    }

    hr = pAudioClient->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
        pAudioClient->Release();
        return;
    }

    // 녹음 포맷 설정 (모노, 16bit, 44.1kHz)
    pwfx->wFormatTag = WAVE_FORMAT_PCM;
    pwfx->nChannels = channels;
    pwfx->nSamplesPerSec = sampleRate;
    pwfx->wBitsPerSample = bitsPerSample;
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
                                  AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                  hnsRequestedDuration, 0, pwfx, NULL);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client for mic " << index << std::endl;
        CoTaskMemFree(pwfx);
        pAudioClient->Release();
        return;
    }

    hr = pAudioClient->GetService(IID_PPV_ARGS(&pCaptureClient));
    if (FAILED(hr)) {
        CoTaskMemFree(pwfx);
        pAudioClient->Release();
        return;
    }

    pAudioClient->Start();

    std::vector<BYTE> buffer;
    std::ofstream wavFile(filename, std::ios::binary);
    if (!wavFile.is_open()) {
        std::cerr << "Failed to open audio output file: " << filename << std::endl;
        pAudioClient->Stop();
        CoTaskMemFree(pwfx);
        pCaptureClient->Release();
        pAudioClient->Release();
        return;
    }

    int totalSamples = sampleRate * durationSec;
    buffer.reserve(totalSamples * pwfx->nBlockAlign);

    // WAV 헤더 임시로 작성 (나중에 실제 데이터 크기로 업데이트)
    WriteWavHeader(wavFile, totalSamples);
    std::streampos dataStart = wavFile.tellp();

    auto start = std::chrono::steady_clock::now();
    std::cout << "Audio recording started for mic " << index << "...\n";

    while (isRecording) {
        UINT32 packetLength = 0;
        pCaptureClient->GetNextPacketSize(&packetLength);
        while (packetLength) {
            BYTE* pData;
            UINT32 numFrames;
            DWORD flags;
            pCaptureClient->GetBuffer(&pData, &numFrames, &flags, NULL, NULL);

            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                size_t bytesToCopy = numFrames * pwfx->nBlockAlign;
                wavFile.write(reinterpret_cast<const char*>(pData), bytesToCopy);
            }

            pCaptureClient->ReleaseBuffer(numFrames);
            pCaptureClient->GetNextPacketSize(&packetLength);
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= durationSec) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 실제 데이터 크기로 WAV 헤더 업데이트
    std::streampos dataEnd = wavFile.tellp();
    int actualDataSize = static_cast<int>(dataEnd - dataStart);
    int actualRiffSize = 36 + actualDataSize;
    
    wavFile.seekp(4);
    wavFile.write(reinterpret_cast<const char*>(&actualRiffSize), 4);
    wavFile.seekp(40);
    wavFile.write(reinterpret_cast<const char*>(&actualDataSize), 4);
    
    wavFile.close();

    pAudioClient->Stop();
    CoTaskMemFree(pwfx);
    pCaptureClient->Release();
    pAudioClient->Release();
    
    std::cout << "Audio recording completed for mic " << index << " and saved to: " << filename << std::endl;
}

void AudioRecorder::RecordingLoop(const std::string& baseFilename, int durationSec) {
    UINT count;
    pDevices->GetCount(&count);

    std::vector<std::thread> threads;
    
    for (UINT i = 0; i < count; ++i) {
        IMMDevice* pDevice = nullptr;
        pDevices->Item(i, &pDevice);
        
        std::string filename = baseFilename + "_mic" + std::to_string(i) + ".wav";
        threads.emplace_back(&AudioRecorder::RecordMic, this, pDevice, i, filename, durationSec);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    isRecording = false;
}

void AudioRecorder::Cleanup() {
    if (pDevices) {
        pDevices->Release();
        pDevices = nullptr;
    }
    if (pEnumerator) {
        pEnumerator->Release();
        pEnumerator = nullptr;
    }
    CoUninitialize();
}