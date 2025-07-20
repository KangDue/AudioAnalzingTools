#include "FanController.h"
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

void printUsage() {
    std::cout << "\n=== Fan Control Utility ===\n";
    std::cout << "Usage:\n";
    std::cout << "  fanctl info                    - Show fan information\n";
    std::cout << "  fanctl set <fanId> <level>     - Set fan level (1-5)\n";
    std::cout << "  fanctl auto <fanId>            - Set fan to auto mode\n";
    std::cout << "  fanctl help                    - Show this help\n";
    std::cout << "\nExamples:\n";
    std::cout << "  fanctl info                    - Display all fan status\n";
    std::cout << "  fanctl set 1 3                - Set fan 1 to level 3\n";
    std::cout << "  fanctl auto 1                 - Set fan 1 to auto mode\n";
    std::cout << "\nNote: This program requires administrator privileges.\n";
}

void printFanInfo(const std::vector<FanInfo>& fans) {
    if (fans.empty()) {
        std::cout << "No fans detected or accessible.\n";
        return;
    }
    
    std::cout << "\n=== Fan Information ===\n";
    std::cout << std::left << std::setw(6) << "Fan ID" 
              << std::setw(8) << "Mode" 
              << std::setw(8) << "Level" 
              << std::setw(8) << "RPM" 
              << std::setw(12) << "Max Level"
              << "Status\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& fan : fans) {
        std::cout << std::left << std::setw(6) << fan.id
                  << std::setw(8) << (fan.autoMode ? "AUTO" : "MANUAL")
                  << std::setw(8) << fan.currentLevel
                  << std::setw(8) << fan.currentRpm
                  << std::setw(12) << fan.maxLevel
                  << fan.status << "\n";
    }
    std::cout << "\n";
}

bool isRunningAsAdmin() {
    BOOL isAdmin = FALSE;
    PSID adminGroup = NULL;
    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    
    if (AllocateAndInitializeSid(&ntAuthority, 2, SECURITY_BUILTIN_DOMAIN_RID,
                                 DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &adminGroup)) {
        CheckTokenMembership(NULL, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }
    
    return isAdmin != FALSE;
}

int main(int argc, char** argv) {
    std::cout << "Fan Control Utility v1.0\n";
    std::cout << "Supports Intel/AMD/Qualcomm Windows laptops\n";
    
    // 관리자 권한 확인
    if (!isRunningAsAdmin()) {
        std::cout << "\nWARNING: This program is not running with administrator privileges.\n";
        std::cout << "Some features may not work properly. Please run as administrator.\n\n";
    }
    
    if (argc < 2) {
        printUsage();
        return 0;
    }
    
    std::string command = argv[1];
    
    if (command == "help" || command == "-h" || command == "--help") {
        printUsage();
        return 0;
    }
    
    // FanController 초기화
    FanController controller;
    std::cout << "\nInitializing fan controller...\n";
    
    if (!controller.Initialize()) {
        std::cerr << "Error: Failed to initialize fan controller.\n";
        std::cerr << "Reason: " << controller.GetLastError() << "\n";
        std::cerr << "\nTroubleshooting tips:\n";
        std::cerr << "1. Make sure you're running as administrator\n";
        std::cerr << "2. Check if your laptop supports fan control\n";
        std::cerr << "3. Try updating your BIOS/UEFI firmware\n";
        return 1;
    }
    
    if (command == "info") {
        auto fans = controller.GetFanInfo();
        printFanInfo(fans);
        
        if (fans.empty()) {
            std::cout << "No fans were detected. This could mean:\n";
            std::cout << "- Your laptop doesn't support fan control\n";
            std::cout << "- The fan control interface is not accessible\n";
            std::cout << "- You need administrator privileges\n";
        }
    }
    else if (command == "set" && argc >= 4) {
        int fanId = std::atoi(argv[2]);
        int level = std::atoi(argv[3]);
        
        if (level < 1 || level > 5) {
            std::cerr << "Error: Fan level must be between 1 and 5.\n";
            return 1;
        }
        
        std::cout << "Setting fan " << fanId << " to level " << level << "...\n";
        
        if (controller.SetFanLevel(fanId, level)) {
            std::cout << "Success: Fan " << fanId << " set to level " << level << ".\n";
            
            // 설정 후 상태 확인
            auto fans = controller.GetFanInfo();
            for (const auto& fan : fans) {
                if (fan.id == fanId) {
                    std::cout << "Current status: Level " << fan.currentLevel 
                              << ", RPM " << fan.currentRpm 
                              << ", Mode " << (fan.autoMode ? "AUTO" : "MANUAL") << "\n";
                    break;
                }
            }
        } else {
            std::cerr << "Error: Failed to set fan level.\n";
            std::cerr << "Reason: " << controller.GetLastError() << "\n";
            return 1;
        }
    }
    else if (command == "auto" && argc >= 3) {
        int fanId = std::atoi(argv[2]);
        
        std::cout << "Setting fan " << fanId << " to auto mode...\n";
        
        if (controller.SetAutoMode(fanId)) {
            std::cout << "Success: Fan " << fanId << " set to auto mode.\n";
            
            // 설정 후 상태 확인
            auto fans = controller.GetFanInfo();
            for (const auto& fan : fans) {
                if (fan.id == fanId) {
                    std::cout << "Current status: Level " << fan.currentLevel 
                              << ", RPM " << fan.currentRpm 
                              << ", Mode " << (fan.autoMode ? "AUTO" : "MANUAL") << "\n";
                    break;
                }
            }
        } else {
            std::cerr << "Error: Failed to set auto mode.\n";
            std::cerr << "Reason: " << controller.GetLastError() << "\n";
            return 1;
        }
    }
    else {
        std::cerr << "Error: Unknown command '" << command << "'\n";
        printUsage();
        return 1;
    }
    
    return 0;
}