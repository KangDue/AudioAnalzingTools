#include "FanController.h"
#include <sstream>
#include <iomanip>

// IOCTL 코드 정의
#define IOCTL_EC_READ_MEMORY    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_EC_WRITE_MEMORY   CTL_CODE(FILE_DEVICE_UNKNOWN, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)

struct EC_MEMORY_REQUEST {
    BYTE Address;
    BYTE Data;
};

FanController::FanController() {
    CoInitializeEx(0, COINIT_MULTITHREADED);
}

FanController::~FanController() {
    cleanupWmi();
    cleanupEcAccess();
    CoUninitialize();
}

bool FanController::Initialize() {
    lastError.clear();
    
    // WMI 초기화 시도
    if (initWmi()) {
        wmiAvailable = true;
        std::cout << "WMI interface initialized successfully.\n";
    } else {
        std::cout << "WMI interface initialization failed, trying EC access...\n";
    }
    
    // EC 직접 접근 초기화 시도
    if (initEcAccess()) {
        ecAvailable = true;
        std::cout << "EC direct access initialized successfully.\n";
    } else {
        std::cout << "EC direct access initialization failed.\n";
    }
    
    initialized = (wmiAvailable || ecAvailable);
    
    if (!initialized) {
        lastError = "Neither WMI nor EC access methods are available";
        return false;
    }
    
    return true;
}

bool FanController::initWmi() {
    HRESULT hr = CoInitializeSecurity(
        NULL, -1, NULL, NULL,
        RPC_C_AUTHN_LEVEL_DEFAULT,
        RPC_C_IMP_LEVEL_IMPERSONATE,
        NULL, EOAC_NONE, NULL
    );
    
    if (FAILED(hr) && hr != RPC_E_TOO_LATE) {
        return false;
    }
    
    hr = CoCreateInstance(
        CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER,
        IID_IWbemLocator, (LPVOID*)&pLoc
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    hr = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\WMI"), NULL, NULL, 0, NULL, 0, 0, &pSvc
    );
    
    if (FAILED(hr)) {
        if (pLoc) {
            pLoc->Release();
            pLoc = nullptr;
        }
        return false;
    }
    
    hr = CoSetProxyBlanket(
        pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL,
        RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
        NULL, EOAC_NONE
    );
    
    return SUCCEEDED(hr);
}

void FanController::cleanupWmi() {
    if (pSvc) {
        pSvc->Release();
        pSvc = nullptr;
    }
    if (pLoc) {
        pLoc->Release();
        pLoc = nullptr;
    }
}

bool FanController::initEcAccess() {
    // EC 장치에 접근 시도
    hEcDevice = CreateFileA(
        "\\\\.\\ACPIEC",
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    if (hEcDevice == INVALID_HANDLE_VALUE) {
        // 대안 경로 시도
        hEcDevice = CreateFileA(
            "\\\\.\\EC",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
    }
    
    return (hEcDevice != INVALID_HANDLE_VALUE);
}

void FanController::cleanupEcAccess() {
    if (hEcDevice != INVALID_HANDLE_VALUE) {
        CloseHandle(hEcDevice);
        hEcDevice = INVALID_HANDLE_VALUE;
    }
}

std::vector<FanInfo> FanController::GetFanInfo() {
    std::vector<FanInfo> fans;
    
    if (!initialized) {
        return fans;
    }
    
    // WMI 방식 먼저 시도
    if (wmiAvailable && queryFanDataWmi(fans)) {
        return fans;
    }
    
    // EC 직접 접근 방식 시도
    if (ecAvailable && queryFanDataEc(fans)) {
        return fans;
    }
    
    lastError = "Failed to retrieve fan information";
    return fans;
}

bool FanController::SetFanLevel(int fanId, int level) {
    if (!initialized || level < 1 || level > 5) {
        lastError = "Invalid parameters or not initialized";
        return false;
    }
    
    // WMI 방식 먼저 시도
    if (wmiAvailable && callFanMethodWmi(fanId, level, false)) {
        return true;
    }
    
    // EC 직접 접근 방식 시도
    if (ecAvailable && callFanMethodEc(fanId, level, false)) {
        return true;
    }
    
    lastError = "Failed to set fan level";
    return false;
}

bool FanController::SetAutoMode(int fanId) {
    if (!initialized) {
        lastError = "Not initialized";
        return false;
    }
    
    // WMI 방식 먼저 시도
    if (wmiAvailable && callFanMethodWmi(fanId, 0, true)) {
        return true;
    }
    
    // EC 직접 접근 방식 시도
    if (ecAvailable && callFanMethodEc(fanId, 0, true)) {
        return true;
    }
    
    lastError = "Failed to set auto mode";
    return false;
}

bool FanController::queryFanDataWmi(std::vector<FanInfo>& fans) {
    if (!pSvc) return false;
    
    // 다양한 OEM WMI 클래스 시도
    std::vector<std::string> wmiClasses = {
        "SELECT * FROM MSAcpi_ThermalZoneTemperature",
        "SELECT * FROM Win32_Fan",
        "SELECT * FROM CIM_Fan",
        "SELECT * FROM LENOVO_FanInformation",
        "SELECT * FROM DELL_FanSensor",
        "SELECT * FROM HP_FanSpeed"
    };
    
    for (const auto& query : wmiClasses) {
        IEnumWbemClassObject* pEnumerator = nullptr;
        HRESULT hr = pSvc->ExecQuery(
            bstr_t("WQL"),
            bstr_t(query.c_str()),
            WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
            NULL, &pEnumerator
        );
        
        if (SUCCEEDED(hr) && pEnumerator) {
            IWbemClassObject* pObj = nullptr;
            ULONG returned = 0;
            
            while (SUCCEEDED(pEnumerator->Next(WBEM_INFINITE, 1, &pObj, &returned)) && returned) {
                FanInfo info{};
                info.id = static_cast<int>(fans.size());
                info.currentRpm = 0;
                info.currentLevel = 1;
                info.autoMode = true;
                info.maxLevel = 5;
                info.status = "WMI Detected";
                
                fans.push_back(info);
                pObj->Release();
            }
            
            pEnumerator->Release();
            if (!fans.empty()) {
                return true;
            }
        }
    }
    
    return false;
}

bool FanController::callFanMethodWmi(int fanId, int level, bool autoMode) {
    // WMI 메서드 호출 구현
    // 실제 구현은 OEM별로 다르므로 기본 구조만 제공
    return false;
}

bool FanController::queryFanDataEc(std::vector<FanInfo>& fans) {
    if (hEcDevice == INVALID_HANDLE_VALUE) return false;
    
    // Fan 1 정보 읽기
    BYTE fan1RpmLow, fan1RpmHigh, fan1Control;
    if (readEcRegister(EC_FAN1_RPM_LOW, fan1RpmLow) &&
        readEcRegister(EC_FAN1_RPM_HIGH, fan1RpmHigh) &&
        readEcRegister(EC_FAN1_CONTROL, fan1Control)) {
        
        FanInfo fan1{};
        fan1.id = 1;
        fan1.currentRpm = (fan1RpmHigh << 8) | fan1RpmLow;
        fan1.currentLevel = (fan1Control & 0x0F);
        fan1.autoMode = (fan1Control & 0x80) != 0;
        fan1.maxLevel = 5;
        fan1.status = "EC Direct Access";
        fans.push_back(fan1);
    }
    
    // Fan 2 정보 읽기 (있는 경우)
    BYTE fan2RpmLow, fan2RpmHigh, fan2Control;
    if (readEcRegister(EC_FAN2_RPM_LOW, fan2RpmLow) &&
        readEcRegister(EC_FAN2_RPM_HIGH, fan2RpmHigh) &&
        readEcRegister(EC_FAN2_CONTROL, fan2Control)) {
        
        FanInfo fan2{};
        fan2.id = 2;
        fan2.currentRpm = (fan2RpmHigh << 8) | fan2RpmLow;
        fan2.currentLevel = (fan2Control & 0x0F);
        fan2.autoMode = (fan2Control & 0x80) != 0;
        fan2.maxLevel = 5;
        fan2.status = "EC Direct Access";
        fans.push_back(fan2);
    }
    
    return !fans.empty();
}

bool FanController::callFanMethodEc(int fanId, int level, bool autoMode) {
    if (hEcDevice == INVALID_HANDLE_VALUE) return false;
    
    BYTE controlRegister = (fanId == 1) ? EC_FAN1_CONTROL : EC_FAN2_CONTROL;
    BYTE controlValue;
    
    if (!readEcRegister(controlRegister, controlValue)) {
        return false;
    }
    
    if (autoMode) {
        // Auto 모드 설정
        controlValue |= 0x80;  // Auto 비트 설정
    } else {
        // Manual 모드 설정
        controlValue &= 0x70;  // Auto 비트 클리어, 레벨 비트 클리어
        controlValue |= (level & 0x0F);  // 레벨 설정
    }
    
    return writeEcRegister(controlRegister, controlValue);
}

bool FanController::readEcRegister(BYTE address, BYTE& value) {
    if (hEcDevice == INVALID_HANDLE_VALUE) return false;
    
    EC_MEMORY_REQUEST request;
    request.Address = address;
    request.Data = 0;
    
    DWORD bytesReturned;
    BOOL result = DeviceIoControl(
        hEcDevice,
        IOCTL_EC_READ_MEMORY,
        &request,
        sizeof(request),
        &request,
        sizeof(request),
        &bytesReturned,
        NULL
    );
    
    if (result) {
        value = request.Data;
        return true;
    }
    
    return false;
}

bool FanController::writeEcRegister(BYTE address, BYTE value) {
    if (hEcDevice == INVALID_HANDLE_VALUE) return false;
    
    EC_MEMORY_REQUEST request;
    request.Address = address;
    request.Data = value;
    
    DWORD bytesReturned;
    BOOL result = DeviceIoControl(
        hEcDevice,
        IOCTL_EC_WRITE_MEMORY,
        &request,
        sizeof(request),
        NULL,
        0,
        &bytesReturned,
        NULL
    );
    
    return (result != FALSE);
}