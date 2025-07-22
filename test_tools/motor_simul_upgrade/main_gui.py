import customtkinter as ctk
import math
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import time
from motor_simulation import BLDCMotorSimulation

class MotorSimulationGUI:
    def __init__(self):
        # CustomTkinter 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 메인 윈도우
        self.root = ctk.CTk()
        self.root.title("BLDC Motor D-Q Control Simulation")
        self.root.geometry("1400x900")
        
        # 모터 시뮬레이션 인스턴스
        self.motor = BLDCMotorSimulation()
        
        # 시뮬레이션 상태
        self.is_running = False
        self.simulation_thread = None
        
        # GUI 구성
        self.setup_gui()
        
        # 기본 파라미터 적용
        self.apply_parameters()
        
        # 기본 신호 설정
        self.update_d_signal()
        
    def setup_gui(self):
        """GUI 레이아웃 설정"""
        # 메인 프레임
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 좌측 제어 패널
        self.setup_control_panel(main_frame)
        
        # 우측 시각화 패널
        self.setup_visualization_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """제어 패널 설정"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        
        # 제목
        title_label = ctk.CTkLabel(control_frame, text="Motor Control Panel", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=20)
        
        # 모터 파라미터 섹션
        self.setup_motor_parameters(control_frame)
        
        # D축 신호 설정 섹션
        self.setup_d_signal_controls(control_frame)
        
        # 시뮬레이션 제어 섹션
        self.setup_simulation_controls(control_frame)
        
        # 상태 표시 섹션
        self.setup_status_display(control_frame)
        
    def setup_motor_parameters(self, parent):
        """모터 파라미터 설정"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(param_frame, text="Motor Parameters", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # 극수 설정
        poles_frame = ctk.CTkFrame(param_frame)
        poles_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(poles_frame, text="Poles:").pack(side="left")
        self.poles_var = ctk.StringVar(value="4")
        poles_entry = ctk.CTkEntry(poles_frame, textvariable=self.poles_var, width=80)
        poles_entry.pack(side="right")
        
        # PWM 주파수 설정
        pwm_frame = ctk.CTkFrame(param_frame)
        pwm_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(pwm_frame, text="PWM Freq (Hz):").pack(side="left")
        self.pwm_freq_var = ctk.StringVar(value="30000")
        pwm_entry = ctk.CTkEntry(pwm_frame, textvariable=self.pwm_freq_var, width=80)
        pwm_entry.pack(side="right")
        
        # 전류각 변화량 설정
        angle_frame = ctk.CTkFrame(param_frame)
        angle_frame.pack(fill="x", padx=10, pady=5)
        
        # 단위 선택
        unit_frame = ctk.CTkFrame(angle_frame)
        unit_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(unit_frame, text="Current Angle Δ:").pack(side="left")
        self.angle_unit_var = ctk.StringVar(value="degree")
        ctk.CTkRadioButton(unit_frame, text="deg", variable=self.angle_unit_var, 
                          value="degree", width=50).pack(side="right", padx=2)
        ctk.CTkRadioButton(unit_frame, text="rad", variable=self.angle_unit_var, 
                          value="radian", width=50).pack(side="right", padx=2)
        
        # 값 입력
        value_frame = ctk.CTkFrame(angle_frame)
        value_frame.pack(fill="x", pady=2)
        self.angle_increment_var = ctk.StringVar(value="5.73")  # ~0.1 rad in degrees
        angle_entry = ctk.CTkEntry(value_frame, textvariable=self.angle_increment_var, width=80)
        angle_entry.pack(side="right")
        self.angle_unit_label = ctk.CTkLabel(value_frame, text="(degree)")
        self.angle_unit_label.pack(side="right", padx=5)
        
        # 파라미터 적용 버튼
        apply_btn = ctk.CTkButton(param_frame, text="Apply Parameters", 
                                 command=self.apply_parameters)
        apply_btn.pack(pady=10)
        
    def setup_d_signal_controls(self, parent):
        """D-Q축 신호 제어 설정"""
        signal_frame = ctk.CTkFrame(parent)
        signal_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(signal_frame, text="D-Q Axis Signals", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # 탭 생성
        self.signal_tabview = ctk.CTkTabview(signal_frame)
        self.signal_tabview.pack(fill="x", padx=10, pady=5)
        
        # D축 탭
        self.signal_tabview.add("D-axis")
        self.setup_axis_signal_controls(self.signal_tabview.tab("D-axis"), "d")
        
        # Q축 탭
        self.signal_tabview.add("Q-axis")
        self.setup_axis_signal_controls(self.signal_tabview.tab("Q-axis"), "q")
        
    def setup_axis_signal_controls(self, parent, axis):
         """개별 축 신호 제어 설정"""
         # 축별 변수 초기화
         if axis == "d":
             self.d_signal_type_var = ctk.StringVar(value="sine")
             self.d_amplitude_var = ctk.StringVar(value="1.0")
             self.d_frequency_var = ctk.StringVar(value="10.0")
             self.d_custom_formula_var = ctk.StringVar(value="np.sin(2*np.pi*t) + 0.5*np.cos(4*np.pi*t)")
             signal_type_var = self.d_signal_type_var
             amplitude_var = self.d_amplitude_var
             frequency_var = self.d_frequency_var
             custom_formula_var = self.d_custom_formula_var
             update_command = self.update_d_signal
         else:  # q축
             self.q_signal_type_var = ctk.StringVar(value="sine")
             self.q_amplitude_var = ctk.StringVar(value="1.0")
             self.q_frequency_var = ctk.StringVar(value="5.0")
             self.q_custom_formula_var = ctk.StringVar(value="np.cos(2*np.pi*t)")
             signal_type_var = self.q_signal_type_var
             amplitude_var = self.q_amplitude_var
             frequency_var = self.q_frequency_var
             custom_formula_var = self.q_custom_formula_var
             update_command = self.update_q_signal
         
         # 신호 타입 선택
         signal_type_frame = ctk.CTkFrame(parent)
         signal_type_frame.pack(fill="x", padx=10, pady=5)
         
         ctk.CTkRadioButton(signal_type_frame, text="Sine", variable=signal_type_var, 
                           value="sine", command=update_command).pack(anchor="w")
         ctk.CTkRadioButton(signal_type_frame, text="Cosine", variable=signal_type_var, 
                           value="cosine", command=update_command).pack(anchor="w")
         ctk.CTkRadioButton(signal_type_frame, text="Custom", variable=signal_type_var, 
                           value="custom", command=update_command).pack(anchor="w")
        
        # 신호 파라미터
         # 진폭
         amp_frame = ctk.CTkFrame(parent)
         amp_frame.pack(fill="x", padx=10, pady=5)
         ctk.CTkLabel(amp_frame, text="Amplitude:").pack(side="left")
         amp_entry = ctk.CTkEntry(amp_frame, textvariable=amplitude_var, width=80)
         amp_entry.pack(side="right")
         
         # 주파수
         freq_frame = ctk.CTkFrame(parent)
         freq_frame.pack(fill="x", padx=10, pady=5)
         ctk.CTkLabel(freq_frame, text="Frequency (Hz):").pack(side="left")
         freq_entry = ctk.CTkEntry(freq_frame, textvariable=frequency_var, width=80)
         freq_entry.pack(side="right")
         
         # 커스텀 수식
         custom_frame = ctk.CTkFrame(parent)
         custom_frame.pack(fill="x", padx=10, pady=5)
         ctk.CTkLabel(custom_frame, text="Custom Formula:").pack(anchor="w")
         custom_entry = ctk.CTkEntry(custom_frame, textvariable=custom_formula_var, width=250)
         custom_entry.pack(fill="x", padx=5, pady=5)
         
         # 신호 업데이트 버튼
         update_signal_btn = ctk.CTkButton(parent, text=f"Update {axis.upper()}-axis Signal", 
                                          command=update_command)
         update_signal_btn.pack(pady=10)
        
    def setup_simulation_controls(self, parent):
        """시뮬레이션 제어 설정"""
        sim_frame = ctk.CTkFrame(parent)
        sim_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(sim_frame, text="Simulation Control", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # 시뮬레이션 시간 설정
        time_frame = ctk.CTkFrame(sim_frame)
        time_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(time_frame, text="Duration (s):").pack(side="left")
        self.duration_var = ctk.StringVar(value="1.0")
        time_entry = ctk.CTkEntry(time_frame, textvariable=self.duration_var, width=80)
        time_entry.pack(side="right")
        
        # 제어 버튼들
        button_frame = ctk.CTkFrame(sim_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.start_btn = ctk.CTkButton(button_frame, text="Start Simulation", 
                                      command=self.start_simulation)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(button_frame, text="Stop", 
                                     command=self.stop_simulation, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        self.reset_btn = ctk.CTkButton(button_frame, text="Reset", 
                                      command=self.reset_simulation)
        self.reset_btn.pack(side="left", padx=5)
        
    def setup_status_display(self, parent):
        """상태 표시 설정"""
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(status_frame, text="Status", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # RPM 표시
        self.rpm_label = ctk.CTkLabel(status_frame, text="RPM: 0.0")
        self.rpm_label.pack(pady=5)
        
        # 전류 표시
        self.current_label = ctk.CTkLabel(status_frame, text="Id: 0.0 A, Iq: 0.0 A")
        self.current_label.pack(pady=5)
        
        # 시뮬레이션 상태
        self.status_label = ctk.CTkLabel(status_frame, text="Status: Ready")
        self.status_label.pack(pady=5)
        
    def setup_visualization_panel(self, parent):
        """시각화 패널 설정"""
        viz_frame = ctk.CTkFrame(parent)
        viz_frame.pack(side="right", fill="both", expand=True)
        
        # 시각화 탭뷰 생성
        self.viz_tabview = ctk.CTkTabview(viz_frame)
        self.viz_tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 시뮬레이션 결과 탭
        self.viz_tabview.add("Simulation")
        self.setup_simulation_plots(self.viz_tabview.tab("Simulation"))
        
        # FFT 스펙트럼 탭
        self.viz_tabview.add("FFT Spectrum")
        self.setup_fft_plots(self.viz_tabview.tab("FFT Spectrum"))
        
    def setup_simulation_plots(self, parent):
        """시뮬레이션 결과 플롯 설정"""
        # Matplotlib Figure 생성
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        
        # 서브플롯 생성
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # D-Q 전류
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Radial Flux
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # RPM
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # 모터 형태
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Rotor Flux
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # D-Q 신호
        
        # 캔버스 생성
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 네비게이션 툴바 추가 (시뮬레이션 탭용)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()
        
        # 마우스 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
        
        # matplotlib 상호작용 모드 활성화
        plt.ion()
        
        # 초기 플롯 설정
        self.setup_plots()
        
        # 각 서브플롯에 대해 개별 상호작용 활성화
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_navigate(True)
            ax.format_coord = lambda x, y: f'x={x:.3f}, y={y:.3f}'
        
    def setup_fft_plots(self, parent):
        """FFT 스펙트럼 플롯 설정"""
        # FFT Figure 생성
        self.fft_fig = Figure(figsize=(12, 8), dpi=100)
        self.fft_fig.patch.set_facecolor('#2b2b2b')
        
        # FFT 서브플롯 생성
        self.fft_ax1 = self.fft_fig.add_subplot(2, 3, 1)  # D축 신호 FFT
        self.fft_ax2 = self.fft_fig.add_subplot(2, 3, 2)  # Q축 신호 FFT
        self.fft_ax3 = self.fft_fig.add_subplot(2, 3, 3)  # Radial Flux FFT
        self.fft_ax4 = self.fft_fig.add_subplot(2, 3, 4)  # RPM FFT
        self.fft_ax5 = self.fft_fig.add_subplot(2, 3, 5)  # Rotor Pole Flux FFT
        self.fft_ax6 = self.fft_fig.add_subplot(2, 3, 6)  # Reserved for future use
        
        # FFT 캔버스 생성
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, parent)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # FFT 네비게이션 툴바 추가
        self.fft_toolbar = NavigationToolbar2Tk(self.fft_canvas, parent)
        self.fft_toolbar.update()
        
        # FFT 마우스 이벤트 연결
        self.fft_canvas.mpl_connect('button_press_event', self.on_fft_click)
        self.fft_canvas.mpl_connect('button_press_event', self.on_fft_right_click)
        
        # FFT 플롯 초기 설정
        self.setup_fft_plots_style()
        
        # FFT 서브플롯에 대해 개별 상호작용 활성화
        for ax in [self.fft_ax1, self.fft_ax2, self.fft_ax3, self.fft_ax4, self.fft_ax5, self.fft_ax6]:
            ax.set_navigate(True)
            ax.format_coord = lambda x, y: f'freq={x:.1f}Hz, mag={y:.3f}'
        
    def on_click(self, event):
        """시뮬레이션 플롯 클릭 이벤트 핸들러"""
        if event.inaxes is not None:
            # 클릭된 축을 활성화
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                if ax == event.inaxes:
                    ax.set_facecolor('#2a2a2a')  # 선택된 축 강조
                else:
                    ax.set_facecolor('#1e1e1e')  # 다른 축은 기본색
            self.canvas.draw_idle()
            
    def on_fft_click(self, event):
        """FFT 플롯 클릭 이벤트 핸들러"""
        if event.inaxes is not None:
            # 클릭된 축을 활성화
            for ax in [self.fft_ax1, self.fft_ax2, self.fft_ax3, self.fft_ax4, self.fft_ax5, self.fft_ax6]:
                if ax == event.inaxes:
                    ax.set_facecolor('#2a2a2a')  # 선택된 축 강조
                else:
                    ax.set_facecolor('#1e1e1e')  # 다른 축은 기본색
            self.fft_canvas.draw_idle()
             
    def on_right_click(self, event):
        """시뮬레이션 플롯 우클릭 이벤트 핸들러"""
        if event.button == 3 and event.inaxes is not None:  # 우클릭
            self.show_axis_range_dialog(event.inaxes, self.canvas)
            
    def on_fft_right_click(self, event):
        """FFT 플롯 우클릭 이벤트 핸들러"""
        if event.button == 3 and event.inaxes is not None:  # 우클릭
            self.show_axis_range_dialog(event.inaxes, self.fft_canvas)
            
    def show_axis_range_dialog(self, ax, canvas):
        """축 범위 설정 다이얼로그 표시"""
        try:
            # 현재 축 범위 가져오기
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 다이얼로그 창 생성
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("축 범위 설정")
            dialog.geometry("300x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # X축 범위 설정
            ctk.CTkLabel(dialog, text="X축 범위:").pack(pady=5)
            x_frame = ctk.CTkFrame(dialog)
            x_frame.pack(pady=5, padx=20, fill="x")
            
            ctk.CTkLabel(x_frame, text="최소값:").grid(row=0, column=0, padx=5)
            x_min_entry = ctk.CTkEntry(x_frame, width=80)
            x_min_entry.insert(0, f"{xlim[0]:.3f}")
            x_min_entry.grid(row=0, column=1, padx=5)
            
            ctk.CTkLabel(x_frame, text="최대값:").grid(row=0, column=2, padx=5)
            x_max_entry = ctk.CTkEntry(x_frame, width=80)
            x_max_entry.insert(0, f"{xlim[1]:.3f}")
            x_max_entry.grid(row=0, column=3, padx=5)
            
            # Y축 범위 설정
            ctk.CTkLabel(dialog, text="Y축 범위:").pack(pady=5)
            y_frame = ctk.CTkFrame(dialog)
            y_frame.pack(pady=5, padx=20, fill="x")
            
            ctk.CTkLabel(y_frame, text="최소값:").grid(row=0, column=0, padx=5)
            y_min_entry = ctk.CTkEntry(y_frame, width=80)
            y_min_entry.insert(0, f"{ylim[0]:.3f}")
            y_min_entry.grid(row=0, column=1, padx=5)
            
            ctk.CTkLabel(y_frame, text="최대값:").grid(row=0, column=2, padx=5)
            y_max_entry = ctk.CTkEntry(y_frame, width=80)
            y_max_entry.insert(0, f"{ylim[1]:.3f}")
            y_max_entry.grid(row=0, column=3, padx=5)
            
            # 버튼 프레임
            btn_frame = ctk.CTkFrame(dialog)
            btn_frame.pack(pady=10, padx=20, fill="x")
            
            def apply_range():
                try:
                    x_min = float(x_min_entry.get())
                    x_max = float(x_max_entry.get())
                    y_min = float(y_min_entry.get())
                    y_max = float(y_max_entry.get())
                    
                    ax.set_xlim(x_min, x_max)
                    
                    # 로그 스케일 축인 경우 양수 값만 허용
                    if ax.get_yscale() == 'log':
                        if y_min <= 0:
                            y_min = 1e-6  # 최소 양수 값
                        if y_max <= 0:
                            y_max = 1.0   # 기본 최대값
                    
                    ax.set_ylim(y_min, y_max)
                    canvas.draw()
                    dialog.destroy()
                except ValueError:
                    messagebox.showerror("오류", "올바른 숫자를 입력해주세요.")
            
            def reset_range():
                ax.relim()
                ax.autoscale()
                canvas.draw()
                dialog.destroy()
            
            ctk.CTkButton(btn_frame, text="적용", command=apply_range).pack(side="left", padx=5)
            ctk.CTkButton(btn_frame, text="자동 범위", command=reset_range).pack(side="left", padx=5)
            ctk.CTkButton(btn_frame, text="취소", command=dialog.destroy).pack(side="left", padx=5)
            
        except Exception as e:
            print(f"축 범위 설정 오류: {e}")
         
    def setup_plots(self):
        """플롯 초기 설정"""
        # 다크 테마 설정
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # 각 플롯 제목 설정
        self.ax1.set_title('D-Q Currents')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Current (A)')
        
        self.ax2.set_title('Radial Flux')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Flux (Wb)')
        
        self.ax3.set_title('RPM')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('RPM')
        
        self.ax4.set_title('Motor Geometry')
        self.ax4.set_aspect('equal')
        
        self.ax5.set_title('Rotor Pole Flux')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Flux (Wb)')
        
        self.ax6.set_title('D-Q Signals')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.set_ylabel('Amplitude')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def setup_fft_plots_style(self):
        """FFT 플롯 스타일 설정"""
        # 다크 테마 설정
        for ax in [self.fft_ax1, self.fft_ax2, self.fft_ax3, self.fft_ax4, self.fft_ax5, self.fft_ax6]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # 각 FFT 플롯 제목 설정
        self.fft_ax1.set_title('D-axis Signal FFT')
        self.fft_ax1.set_xlabel('Frequency (Hz)')
        self.fft_ax1.set_ylabel('Magnitude')
        
        self.fft_ax2.set_title('Q-axis Signal FFT')
        self.fft_ax2.set_xlabel('Frequency (Hz)')
        self.fft_ax2.set_ylabel('Magnitude')
        
        self.fft_ax3.set_title('Radial Flux FFT')
        self.fft_ax3.set_xlabel('Frequency (Hz)')
        self.fft_ax3.set_ylabel('Magnitude')
        
        self.fft_ax4.set_title('RPM FFT')
        self.fft_ax4.set_xlabel('Frequency (Hz)')
        self.fft_ax4.set_ylabel('Magnitude')
        
        self.fft_ax5.set_title('Rotor Pole Flux FFT')
        self.fft_ax5.set_xlabel('Frequency (Hz)')
        self.fft_ax5.set_ylabel('Magnitude')
        
        self.fft_ax6.set_title('Reserved')
        self.fft_ax6.set_xlabel('Frequency (Hz)')
        self.fft_ax6.set_ylabel('Magnitude')
        
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()
        
    def update_d_signal(self):
        """D축 신호 업데이트"""
        signal_type = self.d_signal_type_var.get()
        
        try:
            amplitude = float(self.d_amplitude_var.get())
            frequency = float(self.d_frequency_var.get())
            
            if signal_type == "sine":
                self.motor.set_d_signal(lambda t: amplitude * np.sin(2 * np.pi * frequency * t))
            elif signal_type == "cosine":
                self.motor.set_d_signal(lambda t: amplitude * np.cos(2 * np.pi * frequency * t))
            elif signal_type == "custom":
                formula = self.d_custom_formula_var.get()
                # 안전한 수식 평가를 위한 네임스페이스
                safe_dict = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, 
                           "log": np.log, "sqrt": np.sqrt, "pi": np.pi}
                self.motor.set_d_signal(lambda t: eval(formula.replace('t', str(t)), safe_dict))
                
        except Exception as e:
            print(f"D-axis signal update error: {e}")
            # 기본 신호로 설정
            self.motor.set_d_signal(lambda t: np.sin(2 * np.pi * t))
            
    def update_q_signal(self):
        """Q축 신호 업데이트"""
        signal_type = self.q_signal_type_var.get()
        
        try:
            amplitude = float(self.q_amplitude_var.get())
            frequency = float(self.q_frequency_var.get())
            
            if signal_type == "sine":
                self.motor.set_q_signal(lambda t: amplitude * np.sin(2 * np.pi * frequency * t))
            elif signal_type == "cosine":
                self.motor.set_q_signal(lambda t: amplitude * np.cos(2 * np.pi * frequency * t))
            elif signal_type == "custom":
                formula = self.q_custom_formula_var.get()
                # 안전한 수식 평가를 위한 네임스페이스
                safe_dict = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, 
                           "log": np.log, "sqrt": np.sqrt, "pi": np.pi}
                self.motor.set_q_signal(lambda t: eval(formula.replace('t', str(t)), safe_dict))
                
        except Exception as e:
            print(f"Q-axis signal update error: {e}")
            # 기본 신호로 설정
            self.motor.set_q_signal(lambda t: np.cos(2 * np.pi * t))
            
    def apply_parameters(self):
        """모터 파라미터 적용"""
        try:
            poles = int(self.poles_var.get())
            pwm_freq = float(self.pwm_freq_var.get())
            angle_increment = float(self.angle_increment_var.get())
            angle_unit = self.angle_unit_var.get()
            
            # 새로운 모터 인스턴스 생성
            self.motor = BLDCMotorSimulation(poles=poles, pwm_frequency=pwm_freq)
            
            # 각도 단위에 따라 설정
            if angle_unit == "degree":
                self.motor.set_current_angle_increment_deg(angle_increment)
                self.angle_unit_label.configure(text="(degree)")
            else:  # radian
                self.motor.set_current_angle_increment(angle_increment)
                self.angle_unit_label.configure(text="(radian)")
            
            # 신호 재설정
            self.update_d_signal()
            self.update_q_signal()
            
            self.status_label.configure(text="Status: Parameters Applied")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")
            
    def start_simulation(self):
        """시뮬레이션 시작"""
        if not self.is_running:
            # 시뮬레이션 시작 전에 이전 상태 초기화
            self.motor.reset_simulation()
            self.clear_plots()
            
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
            # 시뮬레이션 스레드 시작
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
    def stop_simulation(self):
        """시뮬레이션 중지"""
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Stopped")
        
    def reset_simulation(self):
        """시뮬레이션 리셋"""
        self.stop_simulation()
        self.motor.reset_simulation()
        self.clear_plots()
        self.status_label.configure(text="Status: Reset")
        
    def run_simulation(self):
        """시뮬레이션 실행"""
        try:
            duration = float(self.duration_var.get())
            self.motor.run_simulation(duration)
            
            # 결과 업데이트
            self.root.after(0, self.update_plots)
            self.root.after(0, lambda: self.status_label.configure(text="Status: Completed"))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.status_label.configure(text=f"Error: {error_msg}"))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            
    def update_plots(self):
        """플롯 업데이트"""
        if not self.motor.time_data:
            return
            
        # 데이터 가져오기
        time_data = np.array(self.motor.time_data)
        
        # D-Q 전류 플롯
        self.ax1.clear()
        self.ax1.plot(time_data, self.motor.id_data, 'r-', label='Id', linewidth=2)
        self.ax1.plot(time_data, self.motor.iq_data, 'b-', label='Iq', linewidth=2)
        self.ax1.set_title('D-Q Currents')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Current (A)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Radial Flux 플롯
        self.ax2.clear()
        self.ax2.plot(time_data, self.motor.radial_flux_data, 'g-', linewidth=2)
        self.ax2.set_title('Radial Flux')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Flux (Wb)')
        self.ax2.grid(True, alpha=0.3)
        
        # RPM 플롯
        self.ax3.clear()
        self.ax3.plot(time_data, self.motor.rpm_data, 'm-', linewidth=2)
        self.ax3.set_title('RPM')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('RPM')
        self.ax3.grid(True, alpha=0.3)
        
        # 모터 형태 플롯
        self.ax4.clear()
        theta_stator, stator_outer_r, stator_inner_r, (theta_rotor, rotor_r) = self.motor.get_motor_geometry()
        
        # 극좌표 플롯
        self.ax4.plot(theta_stator, stator_outer_r, 'b-', linewidth=2, label='Stator Outer')
        self.ax4.plot(theta_stator, stator_inner_r, 'b--', linewidth=1, label='Stator Inner')
        self.ax4.plot(theta_rotor, rotor_r, 'r-', linewidth=2, label='Rotor')
        self.ax4.set_title('Motor Geometry')
        self.ax4.set_aspect('equal')
        self.ax4.legend()
        
        # Rotor Pole Flux 플롯
        self.ax5.clear()
        self.ax5.plot(time_data, self.motor.rotor_flux_data, 'c-', linewidth=2)
        self.ax5.set_title('Rotor Pole Flux')
        self.ax5.set_xlabel('Time (s)')
        self.ax5.set_ylabel('Flux (Wb)')
        self.ax5.grid(True, alpha=0.3)
        
        # D-Q 신호 플롯
        self.ax6.clear()
        if self.motor.d_signal_func:
            d_signal_data = [self.motor.d_signal_func(t) for t in time_data]
            self.ax6.plot(time_data, d_signal_data, 'y-', linewidth=2, label='D-axis')
        if self.motor.q_signal_func:
            q_signal_data = [self.motor.q_signal_func(t) for t in time_data]
            self.ax6.plot(time_data, q_signal_data, 'orange', linewidth=2, label='Q-axis')
        self.ax6.set_title('D-Q Signals')
        self.ax6.set_xlabel('Time (s)')
        self.ax6.set_ylabel('Amplitude')
        self.ax6.legend()
        self.ax6.grid(True, alpha=0.3)
        
        # 상태 업데이트
        if self.motor.rpm_data:
            # 기계적 RPM/RPS
            mech_rpm = self.motor.rpm_data[-1]
            mech_rps = mech_rpm / 60.0

            # 전기적 RPM/RPS
            # 전기각속도(rad/s) -> 전기적 RPM/RPS
            elec_rpm = self.motor.omega_electrical * 60 / (2 * math.pi)
            elec_rps = self.motor.omega_electrical / (2 * math.pi)

            current_id = self.motor.id_data[-1]
            current_iq = self.motor.iq_data[-1]

            rpm_text = f"Mech RPM: {mech_rpm:.1f} | Mech RPS: {mech_rps:.2f}\nElec RPM: {elec_rpm:.1f} | Elec RPS: {elec_rps:.2f}"
            self.rpm_label.configure(text=rpm_text)
            self.current_label.configure(text=f"Id: {current_id:.2f} A, Iq: {current_iq:.2f} A")
        
        # 다크 테마 재적용
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # FFT 스펙트럼 업데이트
        self.update_fft_plots()
        
    def update_fft_plots(self):
        """FFT 스펙트럼 플롯 업데이트"""
        if not self.motor.time_data or len(self.motor.time_data) < 10:
            return
            
        # 데이터 가져오기
        time_data = np.array(self.motor.time_data)
        dt = time_data[1] - time_data[0] if len(time_data) > 1 else 1/self.motor.pwm_frequency
        
        # D축 전류 FFT
        self.fft_ax1.clear()
        if self.motor.id_data and len(self.motor.id_data) > 10:
            # 데이터 전처리: 평균 제거
            id_data = np.array(self.motor.id_data)
            id_data = id_data - np.mean(id_data)
            
            # 윈도우 함수 적용
            window = np.hanning(len(id_data))
            id_data_windowed = id_data * window
            
            # FFT 계산
            id_fft = np.fft.fft(id_data_windowed)
            freqs = np.fft.fftfreq(len(id_fft), dt)
            magnitude = np.abs(id_fft)
            
            # 정규화
            magnitude = magnitude * 2.0 / np.sum(window)
            
            # 양의 주파수만 표시, DC 성분 제외
            pos_mask = freqs > 0.1
            valid_magnitude = magnitude[pos_mask]
            valid_freqs = freqs[pos_mask]
            
            # 로그 스케일을 위해 0 이하 값 처리
            if len(valid_magnitude) > 0 and np.max(valid_magnitude) > 0:
                valid_magnitude = np.maximum(valid_magnitude, np.max(valid_magnitude) * 1e-6)
                self.fft_ax1.semilogy(valid_freqs, valid_magnitude, 'r-', linewidth=1)
            else:
                self.fft_ax1.plot(valid_freqs, valid_magnitude, 'r-', linewidth=1)
        self.fft_ax1.set_title('D-axis Current FFT (Log Scale)')
        self.fft_ax1.set_xlabel('Frequency (Hz)')
        self.fft_ax1.set_ylabel('Magnitude (Log)')
        self.fft_ax1.grid(True, alpha=0.3)
        
        # Q축 전류 FFT
        self.fft_ax2.clear()
        if self.motor.iq_data and len(self.motor.iq_data) > 10:
            # 데이터 전처리: 평균 제거
            iq_data = np.array(self.motor.iq_data)
            iq_data = iq_data - np.mean(iq_data)
            
            # 윈도우 함수 적용
            window = np.hanning(len(iq_data))
            iq_data_windowed = iq_data * window
            
            # FFT 계산
            iq_fft = np.fft.fft(iq_data_windowed)
            freqs = np.fft.fftfreq(len(iq_fft), dt)
            magnitude = np.abs(iq_fft)
            
            # 정규화
            magnitude = magnitude * 2.0 / np.sum(window)
            
            # 양의 주파수만 표시, DC 성분 제외
            pos_mask = freqs > 0.1
            valid_magnitude = magnitude[pos_mask]
            valid_freqs = freqs[pos_mask]
            
            # 로그 스케일을 위해 0 이하 값 처리
            if len(valid_magnitude) > 0 and np.max(valid_magnitude) > 0:
                valid_magnitude = np.maximum(valid_magnitude, np.max(valid_magnitude) * 1e-6)
                self.fft_ax2.semilogy(valid_freqs, valid_magnitude, 'b-', linewidth=1)
            else:
                self.fft_ax2.plot(valid_freqs, valid_magnitude, 'b-', linewidth=1)
        self.fft_ax2.set_title('Q-axis Current FFT (Log Scale)')
        self.fft_ax2.set_xlabel('Frequency (Hz)')
        self.fft_ax2.set_ylabel('Magnitude (Log)')
        self.fft_ax2.grid(True, alpha=0.3)
        
        # Radial Flux FFT
        self.fft_ax3.clear()
        if self.motor.radial_flux_data and len(self.motor.radial_flux_data) > 10:
            # 데이터 전처리: 평균 제거 (DC 성분 완전 제거)
            flux_data = np.array(self.motor.radial_flux_data)
            flux_data = flux_data - np.mean(flux_data)
            
            # 윈도우 함수 적용 (스펙트럼 누설 방지)
            window = np.hanning(len(flux_data))
            flux_data_windowed = flux_data * window
            
            # FFT 계산
            flux_fft = np.fft.fft(flux_data_windowed)
            freqs = np.fft.fftfreq(len(flux_fft), dt)
            magnitude = np.abs(flux_fft)
            
            # 정규화 (윈도우 함수 보정)
            magnitude = magnitude * 2.0 / np.sum(window)
            
            # 양의 주파수만 표시, DC 성분 제외
            pos_mask = freqs > 0.1  # 0.1Hz 이하 제외
            valid_magnitude = magnitude[pos_mask]
            valid_freqs = freqs[pos_mask]
            
            # 로그 스케일을 위해 0 이하 값 처리
            if len(valid_magnitude) > 0 and np.max(valid_magnitude) > 0:
                valid_magnitude = np.maximum(valid_magnitude, np.max(valid_magnitude) * 1e-6)
                self.fft_ax3.semilogy(valid_freqs, valid_magnitude, 'g-', linewidth=1)
            else:
                self.fft_ax3.plot(valid_freqs, valid_magnitude, 'g-', linewidth=1)
        self.fft_ax3.set_title('Radial Flux FFT (Log Scale)')
        self.fft_ax3.set_xlabel('Frequency (Hz)')
        self.fft_ax3.set_ylabel('Magnitude (Log)')
        self.fft_ax3.grid(True, alpha=0.3)
        
        # RPM FFT
        self.fft_ax4.clear()
        if self.motor.rpm_data:
            rpm_fft = np.fft.fft(self.motor.rpm_data)
            freqs = np.fft.fftfreq(len(rpm_fft), dt)
            magnitude = np.abs(rpm_fft)
            pos_mask = freqs > 0
            self.fft_ax4.plot(freqs[pos_mask], magnitude[pos_mask], 'm-', linewidth=1)
        self.fft_ax4.set_title('RPM FFT')
        self.fft_ax4.set_xlabel('Frequency (Hz)')
        self.fft_ax4.set_ylabel('Magnitude')
        self.fft_ax4.grid(True, alpha=0.3)
        
        # Rotor Pole Flux FFT
        self.fft_ax5.clear()
        if self.motor.rotor_flux_data and len(self.motor.rotor_flux_data) > 10:
            # 데이터 전처리: 평균 제거 (DC 성분 완전 제거)
            rotor_flux_data = np.array(self.motor.rotor_flux_data)
            rotor_flux_data = rotor_flux_data - np.mean(rotor_flux_data)
            
            # 윈도우 함수 적용 (스펙트럼 누설 방지)
            window = np.hanning(len(rotor_flux_data))
            rotor_flux_data_windowed = rotor_flux_data * window
            
            # FFT 계산
            rotor_flux_fft = np.fft.fft(rotor_flux_data_windowed)
            freqs = np.fft.fftfreq(len(rotor_flux_fft), dt)
            magnitude = np.abs(rotor_flux_fft)
            
            # 정규화 (윈도우 함수 보정)
            magnitude = magnitude * 2.0 / np.sum(window)
            
            # 양의 주파수만 표시, DC 성분 제외
            pos_mask = freqs > 0.1  # 0.1Hz 이하 제외
            valid_magnitude = magnitude[pos_mask]
            valid_freqs = freqs[pos_mask]
            
            # 로그 스케일을 위해 0 이하 값 처리
            if len(valid_magnitude) > 0 and np.max(valid_magnitude) > 0:
                valid_magnitude = np.maximum(valid_magnitude, np.max(valid_magnitude) * 1e-6)
                self.fft_ax5.semilogy(valid_freqs, valid_magnitude, 'c-', linewidth=1)
            else:
                self.fft_ax5.plot(valid_freqs, valid_magnitude, 'c-', linewidth=1)
        self.fft_ax5.set_title('Rotor Pole Flux FFT (Log Scale)')
        self.fft_ax5.set_xlabel('Frequency (Hz)')
        self.fft_ax5.set_ylabel('Magnitude (Log)')
        self.fft_ax5.grid(True, alpha=0.3)
        
        # FFT 플롯 다크 테마 적용
        for ax in [self.fft_ax1, self.fft_ax2, self.fft_ax3, self.fft_ax4, self.fft_ax5, self.fft_ax6]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()
        
    def clear_plots(self):
        """플롯 초기화"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.clear()
        for ax in [self.fft_ax1, self.fft_ax2, self.fft_ax3, self.fft_ax4, self.fft_ax5, self.fft_ax6]:
            ax.clear()
        self.setup_plots()
        self.setup_fft_plots_style()
        
    def run(self):
        """애플리케이션 실행"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MotorSimulationGUI()
    app.run()