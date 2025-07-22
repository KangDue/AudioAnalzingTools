#!/usr/bin/env python3
"""
BLDC Motor D-Q Control Simulation
메인 실행 파일

이 프로그램은 BLDC 모터의 d-q 제어 시뮬레이션을 제공합니다.
- D축 신호 자유 설정 (sin, cos, 커스텀 수식)
- PWM 주파수 기반 시뮬레이션
- Radial flux 시각화
- RPM 및 전류각 제어
- 모터 극수 설정
- 실시간 모터 형태 시각화
- 로터 극 관점의 flux 시뮬레이션
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main_gui import MotorSimulationGUI
except ImportError as e:
    print(f"Import Error: {e}")
    print("필요한 패키지를 설치해주세요:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def main():
    """
    메인 함수
    """
    print("BLDC Motor D-Q Control Simulation 시작...")
    print("GUI 로딩 중...")
    
    try:
        # GUI 애플리케이션 시작
        app = MotorSimulationGUI()
        app.run()
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        sys.exit(1)
    
    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()