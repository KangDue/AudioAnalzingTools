#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include "acc.h"
#include "audio.h"

// 현재 시간을 문자열로 변환 (파일명에 사용)
std::string GetCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

int main(int argc, char* argv[]) {
    std::cout << "=== 가속도계 + 오디오 동시 녹음 시스템 ===\n\n";
    
    // 녹음 시간 설정 (기본값: 10초)
    int durationSec = 10;
    if (argc > 1) {
        try {
            durationSec = std::stoi(argv[1]);
            if (durationSec <= 0) {
                std::cerr << "녹음 시간은 양수여야 합니다.\n";
                return -1;
            }
        } catch (const std::exception& e) {
            std::cerr << "잘못된 녹음 시간 형식입니다. 숫자를 입력해주세요.\n";
            return -1;
        }
    }
    
    std::cout << "녹음 시간: " << durationSec << "초\n\n";
    
    // 파일명에 사용할 타임스탬프 생성
    std::string timestamp = GetCurrentTimeString();
    std::string accFilename = "accel_" + timestamp + ".csv";
    std::string audioBaseFilename = "audio_" + timestamp;
    
    // 가속도계 초기화
    Accelerometer accelerometer;
    if (!accelerometer.Initialize()) {
        std::cerr << "가속도계 초기화 실패\n";
        return -1;
    }
    
    // 오디오 녹음기 초기화
    AudioRecorder audioRecorder;
    if (!audioRecorder.Initialize()) {
        std::cerr << "오디오 녹음기 초기화 실패\n";
        return -1;
    }
    
    // 샘플링 설정
    accelerometer.SetSampleRate(50);  // 50Hz
    audioRecorder.SetAudioFormat(44100, 1, 16);  // 44.1kHz, 모노, 16bit
    
    std::cout << "초기화 완료. 3초 후 녹음을 시작합니다...\n";
    
    // 카운트다운
    for (int i = 3; i > 0; --i) {
        std::cout << i << "... ";
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "\n\n";
    
    // 동시 녹음 시작
    std::cout << "=== 녹음 시작 ===\n";
    auto recordingStart = std::chrono::steady_clock::now();
    
    // 가속도계와 오디오 녹음을 동시에 시작
    accelerometer.Start(accFilename, durationSec);
    audioRecorder.Start(audioBaseFilename, durationSec);
    
    // 진행 상황 표시
    for (int i = 0; i < durationSec; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "녹음 중... " << (i + 1) << "/" << durationSec << "초\n";
    }
    
    // 녹음 완료 대기
    std::cout << "\n녹음 완료 대기 중...\n";
    
    // 두 녹음이 모두 완료될 때까지 대기
    while (accelerometer.IsRecording() || audioRecorder.IsRecording()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    auto recordingEnd = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(recordingEnd - recordingStart).count();
    
    std::cout << "\n=== 녹음 완료 ===\n";
    std::cout << "총 녹음 시간: " << totalTime / 1000.0 << "초\n";
    std::cout << "\n저장된 파일들:\n";
    std::cout << "- 가속도계 데이터: " << accFilename << "\n";
    std::cout << "- 오디오 파일들: " << audioBaseFilename << "_mic*.wav\n";
    
    std::cout << "\n프로그램을 종료합니다.\n";
    
    return 0;
}