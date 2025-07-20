#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <fstream>
#include <chrono>

// Forward declarations
struct IMMDeviceEnumerator;
struct IMMDeviceCollection;
struct IMMDevice;

class AudioRecorder {
public:
    AudioRecorder();
    ~AudioRecorder();

    // 오디오 녹음기 초기화
    bool Initialize();
    
    // 녹음 시작 (기본 파일명, 지속시간)
    void Start(const std::string& baseFilename, int durationSec);
    
    // 녹음 중지
    void Stop();
    
    // 오디오 포맷 설정 (샘플레이트, 채널수, 비트깊이)
    void SetAudioFormat(int sampleRate, int channels, int bitsPerSample);
    
    // 녹음 상태 확인
    bool IsRecording() const;

private:
    IMMDeviceEnumerator* pEnumerator;
    IMMDeviceCollection* pDevices;
    
    std::atomic<bool> isRecording;
    std::thread recordingThread;
    
    int sampleRate;
    int channels;
    int bitsPerSample;
    
    // WAV 헤더 작성
    void WriteWavHeader(std::ofstream& file, int totalSamples);
    
    // 특정 마이크 녹음
    void RecordMic(IMMDevice* pDevice, int index, const std::string& filename, int durationSec);
    
    // 녹음 루프
    void RecordingLoop(const std::string& baseFilename, int durationSec);
    
    // 리소스 정리
    void Cleanup();
};