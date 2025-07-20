#pragma once

#include <string>
#include <thread>
#include <atomic>

// Forward declarations
struct ISensorManager;
struct ISensorCollection;
struct ISensor;

class Accelerometer {
public:
    Accelerometer();
    ~Accelerometer();

    // 가속도계 초기화
    bool Initialize();
    
    // 녹음 시작 (파일명, 지속시간)
    void Start(const std::string& filename, int durationSec);
    
    // 녹음 중지
    void Stop();
    
    // 샘플링 레이트 설정 (Hz)
    void SetSampleRate(int hz);
    
    // 녹음 상태 확인
    bool IsRecording() const;

private:
    ISensorManager* pSensorManager;
    ISensorCollection* pSensors;
    ISensor* pSensor;
    
    std::atomic<bool> isRecording;
    std::thread recordingThread;
    int sampleRateHz;
    
    // 녹음 루프
    void RecordingLoop(const std::string& filename, int durationSec);
    
    // 리소스 정리
    void Cleanup();
};