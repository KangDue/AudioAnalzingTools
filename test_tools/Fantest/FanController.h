#pragma once

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
    std::string status;
};

class FanController {
public:
    FanController();
    ~FanController();

    bool Initialize();
    std::vector<FanInfo> GetFanInfo();
    bool SetFanLevel(int fanId, int level);  // 1~5
    bool SetAutoMode(int fanId);
    bool IsInitialized() const { return initialized; }
    std::string GetLastError() const { return lastError; }

private:
    bool initWmi();
    void cleanupWmi();
    bool initEcAccess();
    void cleanupEcAccess();
    
    // WMI 방식
    bool queryFanDataWmi(std::vector<FanInfo>& fans);
    bool callFanMethodWmi(int fanId, int level, bool autoMode);
    
    // EC 직접 접근 방식
    bool queryFanDataEc(std::vector<FanInfo>& fans);
    bool callFanMethodEc(int fanId, int level, bool autoMode);
    
    // EC 레지스터 접근
    bool readEcRegister(BYTE address, BYTE& value);
    bool writeEcRegister(BYTE address, BYTE value);
    
    IWbemServices* pSvc = nullptr;
    IWbemLocator* pLoc = nullptr;
    HANDLE hEcDevice = INVALID_HANDLE_VALUE;
    
    bool initialized = false;
    bool wmiAvailable = false;
    bool ecAvailable = false;
    std::string lastError;
    
    // EC 메모리 맵 주소 (일반적인 값들)
    static const BYTE EC_FAN1_RPM_LOW = 0x84;
    static const BYTE EC_FAN1_RPM_HIGH = 0x85;
    static const BYTE EC_FAN1_CONTROL = 0x94;
    static const BYTE EC_FAN2_RPM_LOW = 0x86;
    static const BYTE EC_FAN2_RPM_HIGH = 0x87;
    static const BYTE EC_FAN2_CONTROL = 0x95;
    static const BYTE EC_AUTO_MODE = 0x98;
};