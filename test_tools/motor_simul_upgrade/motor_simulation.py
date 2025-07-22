import numpy as np
import math
from typing import Callable, Tuple, List

class BLDCMotorSimulation:
    def __init__(self, poles: int = 4, pwm_frequency: float = 30000.0):
        """
        BLDC 모터 시뮬레이션 클래스
        
        Args:
            poles: 모터 극수 (기본값: 4)
            pwm_frequency: PWM 주파수 Hz (기본값: 30kHz)
        """
        self.poles = poles
        self.pwm_frequency = pwm_frequency
        self.dt = 1.0 / pwm_frequency  # 시간 간격
        
        # 모터 파라미터 (현실적인 값으로 수정)
        self.Ld = 0.001  # d축 인덕턴스 (H)
        self.Lq = 0.001  # q축 인덕턴스 (H)
        self.Rs = 0.1    # 저항 (Ohm)
        self.flux_linkage = 0.1  # 자속쇄교 (Wb)
        self.J = 0.01    # 관성 모멘트 (kg*m^2) - 10배 증가
        self.B = 0.001   # 점성 마찰 계수 - 10배 증가
        
        # 상태 변수
        self.theta_electrical = 0.0  # 전기각 (rad)
        self.theta_mechanical = 0.0  # 기계각 (rad)
        self.omega_electrical = 0.0  # 전기각속도 (rad/s)
        self.omega_mechanical = 0.0  # 기계각속도 (rad/s)
        self.rpm = 0.0
        
        # 전류
        self.id = 0.0  # d축 전류
        self.iq = 0.0  # q축 전류
        
        # 신호 설정
        self.d_signal_func = None
        self.q_signal_func = None
        self.current_angle_increment = 0.0  # 1 tick당 전류각 변화량 (rad)
        self.current_angle_increment_deg = 0.0  # 1 tick당 전류각 변화량 (degree)
        
        # 시뮬레이션 데이터 저장
        self.time_data = []
        self.theta_data = []
        self.id_data = []
        self.iq_data = []
        self.radial_flux_data = []
        self.rpm_data = []
        self.rotor_flux_data = []
        
    def set_d_signal(self, signal_func: Callable[[float], float]):
        """d축 신호 함수 설정"""
        self.d_signal_func = signal_func
        
    def set_q_signal(self, signal_func: Callable[[float], float]):
        """q축 신호 함수 설정"""
        self.q_signal_func = signal_func
        
    def set_current_angle_increment(self, increment: float):
        """1 tick당 전류각 변화량 설정 (rad)"""
        self.current_angle_increment = increment
        self.current_angle_increment_deg = math.degrees(increment)
        
    def set_current_angle_increment_deg(self, increment_deg: float):
        """1 tick당 전류각 변화량 설정 (degree)"""
        self.current_angle_increment_deg = increment_deg
        self.current_angle_increment = math.radians(increment_deg)
        
    def calculate_dq_currents(self, time: float) -> Tuple[float, float]:
        """d-q 전류 계산"""
        # D축 전류 계산
        if self.d_signal_func:
            id_ref = self.d_signal_func(time)
        else:
            id_ref = 0.0
            
        # Q축 전류 계산
        if self.q_signal_func:
            iq_ref = self.q_signal_func(time)
        else:
            # q축 전류는 전류각 변화량에 따라 결정 (기본 동작)
            # current_angle_increment를 적절한 전류로 변환
            iq_amplitude = abs(self.current_angle_increment) * 5.0  # 스케일링 팩터 조정 (10배 감소)
            iq_ref = iq_amplitude
            
        # 간단한 PI 제어기로 전류 제어 시뮬레이션 (이상적인 제어 가정)
        self.id = id_ref
        self.iq = iq_ref
        
        return self.id, self.iq
        
    def calculate_radial_flux(self, time: float) -> float:
        """Radial flux 계산 - DQ inverse park의 alpha 성분만 (AC 성분만, DC 성분 제거)"""
        # D-Q축 자속 계산 (DC 성분인 영구자석 자속 제외)
        flux_d_ac = self.Ld * self.id  # DC 성분 제거
        flux_q_ac = self.Lq * self.iq

        # Park 역변환으로 스테이터 고정 좌표계 자속 계산
        cos_theta_e = math.cos(self.theta_electrical)
        sin_theta_e = math.sin(self.theta_electrical)

        flux_alpha_ac = flux_d_ac * cos_theta_e - flux_q_ac * sin_theta_e
        
        # 방사 방향 자속 = alpha 성분만 사용 (순수 AC 성분만)
        # 기본적인 스테이터 고정 좌표계에서의 radial flux
        radial_flux = flux_alpha_ac
        
        # 기본적인 스테이터 슬롯 효과만 추가 (최소한의 고조파)
        slot_harmonic = 0.02 * math.cos(12 * self.theta_electrical)
        
        radial_flux += slot_harmonic

        return radial_flux
        
    def calculate_rotor_flux_perspective(self, time: float, rotor_pole_angle: float = 0.0) -> float:
        """로터 극 관점에서의 flux 계산 - alpha가 pole과 u,v,w 코일 zone에 따라 기계적 회전하면서 경험하는 radial flux"""
        # D-Q축 자속 계산 (DC 성분인 영구자석 자속 제외)
        flux_d_ac = self.Ld * self.id  # DC 성분 제거
        flux_q_ac = self.Lq * self.iq
        
        # 로터 극의 기계적 위치 (극 오프셋 포함)
        rotor_pole_position = self.theta_mechanical + rotor_pole_angle
        
        # 스테이터 고정 좌표계에서의 자속 (Park 역변환, AC 성분만)
        cos_theta_e = math.cos(self.theta_electrical)
        sin_theta_e = math.sin(self.theta_electrical)
        
        flux_alpha_ac = flux_d_ac * cos_theta_e - flux_q_ac * sin_theta_e
        
        # 로터 극이 기계적 회전하면서 경험하는 기본 radial flux
        # pole 위치에 따른 기본 변조
        cos_rotor = math.cos(rotor_pole_position)
        rotor_flux_base = flux_alpha_ac * cos_rotor
        
        # U,V,W 코일 zone에 따른 radial flux 변동 (로터 극이 회전하면서 경험)
        # pole 수에 따른 공간 고조파 차수 계산
        coil_harmonic_order = self.poles // 2  # pole 쌍 수 기반
        
        # U상 코일 zone 효과 (로터 극 위치에서 경험)
        u_coil_effect = 0.08 * math.cos(coil_harmonic_order * self.theta_electrical) * math.cos(rotor_pole_position)
        
        # V상 코일 zone 효과 (120도 위상차, 로터 극 위치에서 경험)
        v_coil_effect = 0.08 * math.cos(coil_harmonic_order * self.theta_electrical - 2*math.pi/3) * math.cos(rotor_pole_position - 2*math.pi/3)
        
        # W상 코일 zone 효과 (240도 위상차, 로터 극 위치에서 경험)
        w_coil_effect = 0.08 * math.cos(coil_harmonic_order * self.theta_electrical - 4*math.pi/3) * math.cos(rotor_pole_position - 4*math.pi/3)
        
        # 3상 코일 zone 종합 효과
        coil_zone_modulation = u_coil_effect + v_coil_effect + w_coil_effect
        
        # 로터 회전에 따른 추가 공간 고조파 효과
        # 1) 극 쌍 수에 따른 기본 변조
        pole_modulation = 0.05 * math.cos(self.poles * rotor_pole_position)
        
        # 2) 스테이터 슬롯에 의한 고조파 (로터 극이 슬롯을 지나가면서 경험)
        slot_modulation = 0.03 * math.sin(12 * rotor_pole_position)
        
        # 최종 로터 극 자속 (alpha가 기계적 회전하면서 경험하는 총 radial flux)
        rotor_flux = rotor_flux_base + coil_zone_modulation + pole_modulation + slot_modulation
        
        return rotor_flux
        
    def calculate_torque(self) -> float:
        """토크 계산"""
        # 기본 토크 계산 (PMSM 토크 방정식)
        electromagnetic_torque = (3/2) * (self.poles/2) * (self.flux_linkage * self.iq + (self.Ld - self.Lq) * self.id * self.iq)
        
        # current_angle_increment에 따른 속도 제어 토크 (적절한 크기로 조정)
        speed_control_torque = self.current_angle_increment * 0.5  # 속도 명령에 따른 토크 (20배 감소)
        
        # D축 전류에 의한 추가 토크 (자기 저항 토크)
        reluctance_torque = 0.5 * (self.Ld - self.Lq) * self.id * self.id * math.sin(2 * self.theta_electrical)
        
        total_torque = electromagnetic_torque + speed_control_torque + reluctance_torque
        
        return total_torque
        
    def update_mechanical_dynamics(self, torque: float):
        """간단한 RPM 계산 (전기각 변화 속도 기반)"""
        # 전기각 변화 속도 계산 (rad/s)
        # current_angle_increment는 1 PWM tick당 전기각 변화량
        electrical_speed = self.current_angle_increment * self.pwm_frequency  # rad/s
        
        # 기계각 속도 = 전기각 속도 / (poles/2)
        self.omega_mechanical = electrical_speed / (self.poles / 2)
        
        # 전기각 업데이트
        self.omega_electrical = electrical_speed
        self.theta_electrical += self.omega_electrical * self.dt
        
        # 기계각 업데이트
        self.theta_mechanical += self.omega_mechanical * self.dt
        
        # RPM 계산 (기계각 속도 기반)
        self.rpm = self.omega_mechanical * 60 / (2 * math.pi)
        
    def simulate_step(self, time: float):
        """한 스텝 시뮬레이션"""
        # d-q 전류 계산
        self.calculate_dq_currents(time)
        
        # 토크 계산
        torque = self.calculate_torque()
        
        # 기계적 동역학 업데이트
        self.update_mechanical_dynamics(torque)
        
        # Radial flux 계산
        radial_flux = self.calculate_radial_flux(time)
        rotor_flux = self.calculate_rotor_flux_perspective(time)
        
        # 데이터 저장
        self.time_data.append(time)
        self.theta_data.append(self.theta_electrical)
        self.id_data.append(self.id)
        self.iq_data.append(self.iq)
        self.radial_flux_data.append(radial_flux)
        self.rpm_data.append(self.rpm)
        self.rotor_flux_data.append(rotor_flux)
        
    def run_simulation(self, duration: float):
        """시뮬레이션 실행"""
        # 데이터 초기화
        self.time_data.clear()
        self.theta_data.clear()
        self.id_data.clear()
        self.iq_data.clear()
        self.radial_flux_data.clear()
        self.rpm_data.clear()
        self.rotor_flux_data.clear()
        
        # 시뮬레이션 실행
        num_steps = int(duration / self.dt)
        for i in range(num_steps):
            time = i * self.dt
            self.simulate_step(time)
            
    def get_motor_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """모터 기하학적 형태 반환 (스테이터, 로터)"""
        # 스테이터 (외부 원)
        theta_stator = np.linspace(0, 2*np.pi, 100)
        stator_outer_r = np.ones_like(theta_stator) * 1.0
        stator_inner_r = np.ones_like(theta_stator) * 0.7
        
        # 로터 (내부, 극 표현)
        theta_rotor = np.linspace(0, 2*np.pi, 100)
        rotor_r = np.ones_like(theta_rotor) * 0.6
        
        # 로터 극 표현 (돌출부)
        for i in range(self.poles):
            pole_angle = i * 2 * np.pi / self.poles + self.theta_mechanical
            pole_start = pole_angle - np.pi / self.poles / 2
            pole_end = pole_angle + np.pi / self.poles / 2
            
            mask = (theta_rotor >= pole_start) & (theta_rotor <= pole_end)
            rotor_r[mask] = 0.65  # 극 돌출
            
        return theta_stator, stator_outer_r, stator_inner_r, (theta_rotor, rotor_r)
        
    def reset_simulation(self):
        """시뮬레이션 상태 초기화"""
        self.theta_electrical = 0.0
        self.theta_mechanical = 0.0
        self.omega_electrical = 0.0
        self.omega_mechanical = 0.0
        self.rpm = 0.0
        self.id = 0.0
        self.iq = 0.0
        
        self.time_data.clear()
        self.theta_data.clear()
        self.id_data.clear()
        self.iq_data.clear()
        self.radial_flux_data.clear()
        self.rpm_data.clear()
        self.rotor_flux_data.clear()