import sys
import os
import time
import threading
import numpy as np
import cv2
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox,
    QSlider, QCheckBox, QSpinBox, QTextEdit, QProgressBar,
    QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from scipy import signal
from scipy.io import wavfile
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import wave
import struct


class FakeSerialDevice:
    """실제 시리얼 장치가 없을 때 사용하는 가짜 장치"""
    
    def __init__(self):
        self.is_recording = False
        self.recording_start_time = None
        self.sample_rate = 51200  # 51.2kHz로 변경
        self.channels = 32
        self.frame_size = 1926  # 실제 장치와 동일한 프레임 크기
        
    def write(self, data):
        """녹화 명령 처리"""
        if data == b'\x02\x00\x02\x00\x03':
            self.is_recording = True
            self.recording_start_time = time.time()
            print("[Fake Serial] 녹화 시작")
        elif data == b'\x02\x00\x01\x00\x03':
            self.is_recording = False
            print("[Fake Serial] 녹화 종료")
    
    def read(self, size):
        """가짜 오디오 데이터 생성"""
        if self.is_recording and size == self.frame_size:
            # 랜덤 오디오 데이터 생성 (16bit signed)
            frames = size // (self.channels * 2)  # 2 bytes per sample
            fake_data = np.random.randint(-32768, 32767, (frames, self.channels), dtype=np.int16)
            return fake_data.tobytes()
        return b'\x00' * size
    
    def close(self):
        pass


class FakeCameraDevice:
    """실제 카메라가 없을 때 사용하는 가짜 카메라"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.is_opened = True
        
    def read(self):
        """가짜 비디오 프레임 생성 (최적화된 버전)"""
        if not self.is_opened:
            return False, None
            
        # 움직이는 그라디언트 패턴 생성 (벡터화 연산으로 최적화)
        t = self.frame_count / self.fps
        
        # 좌표 그리드 생성
        x_coords, y_coords = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # 벡터화된 색상 계산
        r = (128 + 127 * np.sin(t + x_coords * 0.01)).astype(np.uint8)
        g = (128 + 127 * np.sin(t + y_coords * 0.01 + np.pi/3)).astype(np.uint8)
        b = (128 + 127 * np.sin(t + (x_coords + y_coords) * 0.01 + 2*np.pi/3)).astype(np.uint8)
        
        # BGR 프레임 구성
        frame = np.stack([b, g, r], axis=2)
        
        # 텍스트 오버레이
        cv2.putText(frame, f"FAKE CAMERA - Frame {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.frame_count += 1
        return True, frame
    
    def release(self):
        self.is_opened = False
    
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0
    
    def set(self, prop, value):
        return True


class SerialManager:
    """시리얼 장치 관리자"""
    
    def __init__(self):
        self.device = None
        self.is_fake = False
        self.detect_device()
    
    def detect_device(self):
        """실제 시리얼 장치 감지 또는 가짜 장치 사용"""
        ports = serial.tools.list_ports.comports()
        
        if ports:
            try:
                # 첫 번째 포트로 연결 시도
                test_device = serial.Serial(ports[0].device, 115200, timeout=2)
                
                # 데이터 수신 테스트 (2초 대기)
                test_device.write(b'test\n')  # 테스트 명령 전송
                time.sleep(0.5)
                
                data_received = False
                start_time = time.time()
                while time.time() - start_time < 1.5:  # 1.5초 동안 데이터 확인
                    if test_device.in_waiting > 0:
                        test_device.read(test_device.in_waiting)  # 버퍼 클리어
                        data_received = True
                        break
                    time.sleep(0.1)
                
                if data_received:
                    self.device = test_device
                    self.is_fake = False
                    print(f"[Serial] 실제 장치 연결: {ports[0].device}")
                else:
                    test_device.close()
                    self.device = FakeSerialDevice()
                    self.is_fake = True
                    print("[Serial] 가짜 장치 사용 (데이터 수신 없음)")
                    
            except Exception as e:
                self.device = FakeSerialDevice()
                self.is_fake = True
                print(f"[Serial] 가짜 장치 사용 (연결 실패: {e})")
        else:
            self.device = FakeSerialDevice()
            self.is_fake = True
            print("[Serial] 가짜 장치 사용 (포트 없음)")
    
    def get_status(self):
        return "Fake" if self.is_fake else "Real"
    
    def write(self, data):
        return self.device.write(data)
    
    def read(self, size):
        return self.device.read(size)
    
    def close(self):
        if self.device:
            self.device.close()


class CameraManager:
    """카메라 장치 관리자"""
    
    def __init__(self):
        self.device = None
        self.is_fake = False
        self.detect_device()
    
    def detect_device(self):
        """실제 카메라 감지 또는 가짜 카메라 사용"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.device = cap
                    self.is_fake = False
                    print("[Camera] 실제 카메라 연결 성공")
                else:
                    cap.release()
                    raise Exception("카메라에서 프레임을 읽을 수 없음")
            else:
                raise Exception("카메라를 열 수 없음")
        except:
            self.device = FakeCameraDevice()
            self.is_fake = True
            print("[Camera] 실제 카메라 없음, Fake 모드로 전환")
    
    def get_status(self):
        return "Fake" if self.is_fake else "Real"
    
    def read(self):
        return self.device.read()
    
    def release(self):
        if self.device:
            self.device.release()
    
    def get(self, prop):
        return self.device.get(prop)
    
    def set(self, prop, value):
        return self.device.set(prop, value)


class RecordingThread(QThread):
    """녹화 스레드"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    
    def __init__(self, serial_manager, camera_manager, save_folder, duration=10):
        super().__init__()
        self.serial_manager = serial_manager
        self.camera_manager = camera_manager
        self.save_folder = save_folder
        self.duration = duration
        self.is_recording = False
        
    def run(self):
        self.is_recording = True
        self.status_updated.emit("녹화 시작...")
        
        # 파일 경로 설정
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(self.save_folder, f"audio_{timestamp}.wav")
        video_path = os.path.join(self.save_folder, f"video_{timestamp}.mp4")
        
        # 비디오 설정
        width = int(self.camera_manager.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera_manager.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # 오디오 데이터 저장용
        audio_frames = []
        sample_rate = 51200  # 51.2kHz로 변경
        channels = 32
        
        # Fake 모드인지 확인
        is_fake_mode = self.serial_manager.is_fake or self.camera_manager.is_fake
        
        # Fake 모드일 때는 10초 고정, 실제 모드일 때는 설정된 시간
        actual_duration = 10 if is_fake_mode else self.duration
        
        # 녹화 시작 명령
        self.serial_manager.write(b'\x02\x00\x02\x00\x03')
        
        start_time = time.time()
        frame_count = 0
        total_frames = int(fps * actual_duration)  # 정확한 프레임 수 계산
        
        if is_fake_mode:
            # Fake 모드: 정확히 10초 분량의 데이터 생성 (타이밍 최적화)
            frame_interval = 1.0 / fps
            next_frame_time = start_time
            
            for i in range(total_frames):
                if not self.is_recording:
                    break
                    
                # 진행률 업데이트
                elapsed_sec = int((i / fps))
                self.progress_updated.emit(elapsed_sec)
                self.status_updated.emit(f"녹화 중... {elapsed_sec}/{int(actual_duration)}초")
                
                # 비디오 프레임 캡처
                ret, frame = self.camera_manager.read()
                if ret:
                    video_writer.write(frame)
                
                # 오디오 데이터 읽기 (fake 모드에서는 매 프레임마다 생성)
                audio_data = self.serial_manager.read(1926)
                if len(audio_data) >= 1920:  # 실제 데이터 크기에 맞춰 조정
                    # 1920바이트만 사용 (32채널 * 30샘플 * 2바이트)
                    audio_array = np.frombuffer(audio_data[:1920], dtype=np.int16)
                    audio_array = audio_array.reshape(-1, channels)
                    audio_frames.append(audio_array)
                    print(f"[Audio] 프레임 추가: {len(audio_array)} 샘플")
                
                # 정확한 타이밍 제어
                next_frame_time += frame_interval
                current_time = time.time()
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
        else:
            # 실제 모드: 기존 로직
            while self.is_recording and (time.time() - start_time) < actual_duration:
                elapsed = time.time() - start_time
                elapsed_sec = int(elapsed)
                total_sec = int(actual_duration)
                self.progress_updated.emit(elapsed_sec)
                self.status_updated.emit(f"녹화 중... {elapsed_sec}/{total_sec}초")
                
                # 비디오 프레임 캡처
                ret, frame = self.camera_manager.read()
                if ret:
                    video_writer.write(frame)
                
                # 오디오 데이터 읽기
                try:
                    audio_data = self.serial_manager.read(1926)
                    if len(audio_data) == 1926:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        audio_array = audio_array.reshape(-1, channels)
                        audio_frames.append(audio_array)
                except:
                    pass
                
                time.sleep(1/fps)
        
        # 녹화 종료
        self.serial_manager.write(b'\x02\x00\x01\x00\x03')
        video_writer.release()
        
        # 오디오 파일 저장
        if is_fake_mode:
            # Fake 모드: 정확히 32채널, 51.2kHz, 10초 오디오 생성
            fake_sample_rate = 51200
            fake_channels = 32
            fake_duration = 10
            fake_samples = fake_sample_rate * fake_duration
            
            # 32채널 랜덤 오디오 데이터 생성
            fake_audio = np.random.randint(-32768, 32767, (fake_samples, fake_channels), dtype=np.int16)
            
            # WAV 파일로 저장
            wavfile.write(audio_path, fake_sample_rate, fake_audio)
            print(f"[Audio] Fake 모드 저장 완료: {fake_channels}채널, {fake_sample_rate}Hz, {fake_duration}초")
        elif audio_frames:
            all_audio = np.vstack(audio_frames)
            # WAV 파일로 저장
            wavfile.write(audio_path, sample_rate, all_audio.astype(np.int16))
            print(f"[Audio] 저장 완료: {len(all_audio)} 샘플, {len(all_audio)/sample_rate:.1f}초")
        else:
            print("[Audio] 경고: 오디오 데이터가 없습니다.")
        
        self.progress_updated.emit(int(actual_duration))  # 최종 시간으로 설정
        self.status_updated.emit(f"녹화 완료: {audio_path}, {video_path}")
        self.is_recording = False
    
    def stop_recording(self):
        self.is_recording = False


class RecordTab(QWidget):
    """녹화 탭"""
    
    def __init__(self):
        super().__init__()
        self.serial_manager = SerialManager()
        self.camera_manager = CameraManager()
        self.recording_thread = None
        self.save_folder = ""
        self.init_ui()
        
        # 카메라 화면 설정
        self.setup_camera_display()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 장치 상태 표시
        status_group = QGroupBox("장치 상태")
        status_layout = QGridLayout()
        
        self.serial_status = QLabel(f"Serial: {self.serial_manager.get_status()}")
        self.camera_status = QLabel(f"Camera: {self.camera_manager.get_status()}")
        
        status_layout.addWidget(QLabel("시리얼 장치:"), 0, 0)
        status_layout.addWidget(self.serial_status, 0, 1)
        status_layout.addWidget(QLabel("카메라:"), 1, 0)
        status_layout.addWidget(self.camera_status, 1, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 녹화 설정
        record_group = QGroupBox("녹화 설정")
        record_layout = QGridLayout()
        
        self.folder_btn = QPushButton("저장 폴더 선택")
        self.folder_btn.clicked.connect(self.select_folder)
        self.folder_label = QLabel("폴더가 선택되지 않음")
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(" 초")
        
        record_layout.addWidget(QLabel("저장 폴더:"), 0, 0)
        record_layout.addWidget(self.folder_btn, 0, 1)
        record_layout.addWidget(self.folder_label, 1, 0, 1, 2)
        record_layout.addWidget(QLabel("녹화 시간:"), 2, 0)
        record_layout.addWidget(self.duration_spin, 2, 1)
        
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)
        
        # 녹화 컨트롤
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("녹화 시작")
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn = QPushButton("녹화 중지")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)
        
        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("대기 중...")
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        
        # 카메라 화면 표시 영역
        camera_group = QGroupBox("카메라 화면")
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel("카메라 화면이 여기에 표시됩니다")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        self.setLayout(layout)
        
        # 카메라 화면 업데이트 타이머
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_display)
    
    def setup_camera_display(self):
        """카메라 화면 설정"""
        # 카메라 화면 업데이트 시작 (약 16fps로 최적화)
        self.camera_timer.start(60)  # 약 16fps로 성능 최적화
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if folder:
            self.save_folder = folder
            self.folder_label.setText(f"선택된 폴더: {folder}")
    
    def start_recording(self):
        if not self.save_folder:
            QMessageBox.warning(self, "경고", "저장 폴더를 선택해주세요.")
            return
        
        duration = self.duration_spin.value()
        
        # Fake 모드인지 확인하여 진행률 바 최대값 설정
        is_fake_mode = self.serial_manager.is_fake or self.camera_manager.is_fake
        max_duration = 10 if is_fake_mode else duration
        self.progress_bar.setMaximum(max_duration)
        self.progress_bar.setValue(0)
        
        self.recording_thread = RecordingThread(
            self.serial_manager, self.camera_manager, self.save_folder, duration
        )
        
        self.recording_thread.progress_updated.connect(self.progress_bar.setValue)
        self.recording_thread.status_updated.connect(self.status_label.setText)
        self.recording_thread.finished.connect(self.recording_finished)
        
        self.recording_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def stop_recording(self):
        if self.recording_thread:
            self.recording_thread.stop_recording()
    
    def recording_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def update_camera_display(self):
        """카메라 화면 업데이트 (최적화된 버전)"""
        if self.camera_manager.device:
            ret, frame = self.camera_manager.device.read()
            if ret:
                # 프레임 크기 미리 조정 (성능 최적화)
                label_size = self.camera_label.size()
                target_width = min(640, label_size.width())
                target_height = min(480, label_size.height())
                
                # 프레임 리사이즈 (BGR 상태에서)
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                
                # QImage로 변환
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # QPixmap으로 변환하여 표시 (추가 스케일링 없이)
                pixmap = QPixmap.fromImage(qt_image)
                self.camera_label.setPixmap(pixmap)


class AnalysisTab(QWidget):
    """분석 탭 (Heatmap Overlay)"""
    
    def __init__(self):
        super().__init__()
        self.video_file = None
        self.audio_file = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 파일 로드
        file_group = QGroupBox("파일 로드")
        file_layout = QGridLayout()
        
        self.video_btn = QPushButton("비디오 파일 선택")
        self.audio_btn = QPushButton("오디오 파일 선택")
        self.video_label = QLabel("비디오 파일이 선택되지 않음")
        self.audio_label = QLabel("오디오 파일이 선택되지 않음")
        
        # 파일 선택 이벤트 연결
        self.video_btn.clicked.connect(self.select_video_file)
        self.audio_btn.clicked.connect(self.select_audio_file)
        
        file_layout.addWidget(self.video_btn, 0, 0)
        file_layout.addWidget(self.video_label, 0, 1)
        file_layout.addWidget(self.audio_btn, 1, 0)
        file_layout.addWidget(self.audio_label, 1, 1)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 분석 설정
        analysis_group = QGroupBox("분석 설정")
        analysis_layout = QGridLayout()
        
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_label = QLabel("Alpha: 0.5")
        self.alpha_slider.valueChanged.connect(self.update_alpha_label)
        
        # 컬러맵 선택
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "hot", "cool", "spring", "summer", "autumn", "winter", "jet"])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        
        self.heatmap_only = QCheckBox("Heatmap만 표시")
        self.video_only = QCheckBox("비디오만 표시")
        self.show_colorbar = QCheckBox("컬러바 표시")
        self.show_colorbar.setChecked(True)
        self.show_colorbar.toggled.connect(self.setup_colorbar)
        
        analysis_layout.addWidget(QLabel("Heatmap Alpha:"), 0, 0)
        analysis_layout.addWidget(self.alpha_slider, 0, 1)
        analysis_layout.addWidget(self.alpha_label, 0, 2)
        analysis_layout.addWidget(QLabel("컬러맵:"), 1, 0)
        analysis_layout.addWidget(self.colormap_combo, 1, 1)
        analysis_layout.addWidget(self.show_colorbar, 1, 2)
        analysis_layout.addWidget(self.heatmap_only, 2, 0)
        analysis_layout.addWidget(self.video_only, 2, 1)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # 분석 시작 버튼
        self.analyze_btn = QPushButton("분석 시작")
        self.analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_btn)
        
        # 결과 표시 영역 (비디오 플레이어 + 히트맵 오버레이 + 컬러바)
        result_layout = QHBoxLayout()
        
        self.video_widget = QLabel("분석 결과가 여기에 표시됩니다.")
        self.video_widget.setMinimumSize(640, 480)
        self.video_widget.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        self.video_widget.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.video_widget)
        
        # pyqtgraph 기반 제어 가능한 컬러바
        import pyqtgraph as pg
        self.colorbar_widget = pg.HistogramLUTWidget()
        self.colorbar_widget.setFixedSize(200, 480)
        # 초기에는 ImageItem 설정하지 않음 (나중에 분석 시작 시 설정)
        result_layout.addWidget(self.colorbar_widget)
        
        layout.addLayout(result_layout)
        
        # 재생 컨트롤
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("재생")
        self.pause_btn = QPushButton("일시정지")
        self.stop_btn = QPushButton("정지")
        self.progress_slider = QSlider(Qt.Horizontal)
        
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.stop_btn.clicked.connect(self.stop_video)
        
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.progress_slider)
        
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
        
        # 비디오 재생 관련 변수
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fps = 30
        self.frame_count = 0
        self.total_frames = 0
        self.heatmap_data = None
        self.current_colormap = "viridis"
        self.heatmap_image_item = None
        self.saved_levels = None  # 사용자가 설정한 레벨 저장
        self.levels_connected = False  # 레벨 변경 이벤트 연결 상태
        
        # 초기 컬러바 설정
        self.setup_colorbar()
    
    def select_video_file(self):
        """비디오 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "비디오 파일 선택", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.video_file = file_path
            self.video_label.setText(f"선택됨: {os.path.basename(file_path)}")
    
    def select_audio_file(self):
        """오디오 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "오디오 파일 선택", "", 
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if file_path:
            self.audio_file = file_path
            self.audio_label.setText(f"선택됨: {os.path.basename(file_path)}")
    
    def start_analysis(self):
        """분석 시작"""
        if not self.video_file:
            QMessageBox.warning(self, "경고", "비디오 파일을 선택해주세요.")
            return
            
        try:
            # 비디오 로드
            self.cap = cv2.VideoCapture(self.video_file)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 10초 길이의 랜덤 2D 히트맵 생성 (시간에 따라 변화)
            duration = 10  # 10초 고정
            time_steps = int(self.fps * duration)
            height, width = 480, 640
            
            # 랜덤 히트맵 데이터 생성 (time_steps x height x width)
            self.heatmap_data = np.random.rand(time_steps, height//4, width//4)  # 1/4 크기로 생성 후 업스케일
            
            # 프로그레스 슬라이더 설정
            self.progress_slider.setMaximum(self.total_frames - 1)
            self.progress_slider.setValue(0)
            self.frame_count = 0
            
            # pyqtgraph ImageItem 생성 및 컬러바 연결
            import pyqtgraph as pg
            self.heatmap_image_item = pg.ImageItem()
            self.colorbar_widget.item.setImageItem(self.heatmap_image_item)
            
            # 초기 히트맵 설정
            initial_heatmap = self.heatmap_data[0]
            self.heatmap_image_item.setImage(initial_heatmap)
            
            QMessageBox.information(self, "정보", "분석이 완료되었습니다. 재생 버튼을 눌러주세요.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"분석 중 오류가 발생했습니다: {str(e)}")
    
    def play_video(self):
        """비디오 재생"""
        if self.cap is not None:
            self.timer.start(int(1000 / self.fps))  # fps에 맞춰 타이머 시작
    
    def pause_video(self):
        """비디오 일시정지"""
        self.timer.stop()
    
    def stop_video(self):
        """비디오 정지"""
        self.timer.stop()
        self.frame_count = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.progress_slider.setValue(0)
    
    def update_frame(self):
        """프레임 업데이트 및 히트맵 오버레이"""
        if self.cap is None or self.heatmap_data is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        # 현재 시간에 해당하는 히트맵 인덱스 계산
        time_index = min(int(self.frame_count / self.fps * self.fps), len(self.heatmap_data) - 1)
        
        # 히트맵 생성
        heatmap = self.heatmap_data[time_index]
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # pyqtgraph ImageItem 업데이트 (컬러바와 동기화)
        if self.heatmap_image_item is not None:
            # 현재 레벨 백업 (setImage가 레벨을 변경할 수 있음)
            current_levels = None
            if self.saved_levels is not None:
                current_levels = self.saved_levels
            
            # 자동 레벨 조정 비활성화하고 이미지 설정
            self.heatmap_image_item.setImage(heatmap, autoLevels=False)
            
            # 사용자가 설정한 레벨 복원
            if current_levels is not None:
                self.colorbar_widget.item.setLevels(*current_levels)
        
        # 히트맵을 선택된 컬러맵으로 변환 (비디오 오버레이용)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(self.current_colormap)
        
        # 컬러바 레벨 적용
        if self.saved_levels is not None:
            min_level, max_level = self.saved_levels
            # 히트맵을 컬러바 레벨 범위로 정규화
            heatmap_normalized = np.clip((heatmap_resized - min_level) / (max_level - min_level), 0, 1)
        else:
            heatmap_normalized = heatmap_resized
        
        heatmap_colored = (cmap(heatmap_normalized) * 255).astype(np.uint8)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGBA2BGR)
        
        # 알파 블렌딩으로 오버레이
        alpha = self.alpha_slider.value() / 100.0
        
        if self.heatmap_only.isChecked():
            result_frame = heatmap_colored
        elif self.video_only.isChecked():
            result_frame = frame
        else:
            result_frame = cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)
        
        # QLabel에 표시
        rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 위젯 크기에 맞게 스케일링
        widget_size = self.video_widget.size()
        scaled_image = qt_image.scaled(widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(scaled_image)
        self.video_widget.setPixmap(pixmap)
        
        # 프로그레스 업데이트
        self.frame_count += 1
        self.progress_slider.setValue(self.frame_count)
        
        # 끝에 도달하면 정지
        if self.frame_count >= self.total_frames:
            self.timer.stop()
    
    def update_alpha_label(self):
        """알파 슬라이더 값 업데이트"""
        alpha_value = self.alpha_slider.value() / 100.0
        self.alpha_label.setText(f"Alpha: {alpha_value:.1f}")
    
    def update_colormap(self, colormap_name):
        """컬러맵 업데이트"""
        self.current_colormap = colormap_name
        self.setup_colorbar()
        
    def setup_colorbar(self):
        """pyqtgraph 기반 컬러바 설정"""
        if not self.show_colorbar.isChecked():
            self.colorbar_widget.hide()
            return
        else:
            self.colorbar_widget.show()
            
        try:
            import pyqtgraph as pg
            
            # 컬러맵 매핑 (pyqtgraph 호환)
            colormap_mapping = {
                'viridis': 'viridis',
                'plasma': 'plasma', 
                'inferno': 'inferno',
                'magma': 'magma',
                'hot': 'hot',
                'cool': 'cool',
                'spring': 'spring',
                'summer': 'summer',
                'autumn': 'autumn',
                'winter': 'winter',
                'jet': 'jet'
            }
            
            # 컬러맵 설정
            cmap_name = colormap_mapping.get(self.current_colormap, 'viridis')
            self.colorbar_widget.item.gradient.loadPreset(cmap_name)
            
            # 레벨 범위 설정 (0-1)
            if self.saved_levels is None:
                self.colorbar_widget.item.setLevels(0, 1)
            else:
                self.colorbar_widget.item.setLevels(*self.saved_levels)
            
            # 레벨 변경 이벤트 연결 (중복 연결 방지)
            if not self.levels_connected:
                self.colorbar_widget.item.sigLevelsChanged.connect(self.save_levels)
                self.levels_connected = True
            
        except Exception as e:
            print(f"컬러바 설정 오류: {e}")
    
    def save_levels(self):
        """사용자가 설정한 레벨 저장"""
        try:
            levels = self.colorbar_widget.item.getLevels()
            self.saved_levels = levels
        except Exception as e:
            print(f"레벨 저장 오류: {e}")


class AudioTab(QWidget):
    """오디오 분석 탭"""
    
    def __init__(self):
        super().__init__()
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 파일 로드
        file_group = QGroupBox("WAV 파일 로드")
        file_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("WAV 파일 선택")
        self.file_label = QLabel("파일이 선택되지 않음")
        
        # 파일 선택 이벤트 연결
        self.load_btn.clicked.connect(self.select_audio_file)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 분석 설정
        analysis_group = QGroupBox("분석 설정")
        analysis_layout = QGridLayout()
        
        self.channel_combo = QComboBox()
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["STFT", "FFT", "Waveform"])
        
        analysis_layout.addWidget(QLabel("채널 선택:"), 0, 0)
        analysis_layout.addWidget(self.channel_combo, 0, 1)
        analysis_layout.addWidget(QLabel("분석 방법:"), 1, 0)
        analysis_layout.addWidget(self.analysis_combo, 1, 1)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # 분석 시작 버튼
        self.analyze_btn = QPushButton("분석 시작")
        layout.addWidget(self.analyze_btn)
        
        # 그래프 영역
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # 분석 버튼 이벤트 연결
        self.analyze_btn.clicked.connect(self.analyze_audio)
        
        self.setLayout(layout)
    
    def select_audio_file(self):
        """WAV 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "WAV 파일 선택", "", 
            "WAV Files (*.wav);;All Files (*)"
        )
        if file_path:
            try:
                # WAV 파일 로드
                self.sample_rate, self.audio_data = wavfile.read(file_path)
                self.audio_file = file_path
                self.file_label.setText(f"선택됨: {os.path.basename(file_path)}")
                
                # 채널 콤보박스 업데이트
                self.channel_combo.clear()
                if len(self.audio_data.shape) == 1:
                    self.channel_combo.addItem("Channel 1")
                else:
                    for i in range(self.audio_data.shape[1]):
                        self.channel_combo.addItem(f"Channel {i+1}")
                        
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파일을 로드할 수 없습니다: {str(e)}")
    
    def analyze_audio(self):
        """오디오 분석 수행"""
        if self.audio_data is None:
            QMessageBox.warning(self, "경고", "먼저 WAV 파일을 선택하세요.")
            return
        
        try:
            # 선택된 채널 가져오기
            channel_idx = self.channel_combo.currentIndex()
            if len(self.audio_data.shape) == 1:
                data = self.audio_data
            else:
                data = self.audio_data[:, channel_idx]
            
            # 분석 방법에 따라 처리
            analysis_type = self.analysis_combo.currentText()
            
            self.plot_widget.clear()
            
            if analysis_type == "Waveform":
                # 파형 표시
                time_axis = np.arange(len(data)) / self.sample_rate
                self.plot_widget.plot(time_axis, data, pen='b')
                self.plot_widget.setLabel('left', 'Amplitude')
                self.plot_widget.setLabel('bottom', 'Time (s)')
                
            elif analysis_type == "FFT":
                # FFT 분석
                fft_data = np.fft.fft(data)
                freqs = np.fft.fftfreq(len(data), 1/self.sample_rate)
                magnitude = np.abs(fft_data)
                
                # 양의 주파수만 표시
                pos_mask = freqs >= 0
                self.plot_widget.plot(freqs[pos_mask], magnitude[pos_mask], pen='r')
                self.plot_widget.setLabel('left', 'Magnitude')
                self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
                
            elif analysis_type == "STFT":
                # STFT 분석 (간단한 구현)
                from scipy import signal
                f, t, Zxx = signal.stft(data, self.sample_rate, nperseg=1024)
                
                # 스펙트로그램 표시 (간단한 방법)
                magnitude = np.abs(Zxx)
                self.plot_widget.clear()
                
                # 평균 스펙트럼 표시
                avg_spectrum = np.mean(magnitude, axis=1)
                self.plot_widget.plot(f, avg_spectrum, pen='g')
                self.plot_widget.setLabel('left', 'Average Magnitude')
                self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
                
        except Exception as e:
             QMessageBox.critical(self, "오류", f"분석 중 오류가 발생했습니다: {str(e)}")


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("동시 녹화 PySerial + 카메라 (Fake 지원)")
        self.setGeometry(100, 100, 1200, 800)
        
        # 탭 위젯 생성
        tab_widget = QTabWidget()
        
        # 각 탭 추가
        self.record_tab = RecordTab()
        self.analysis_tab = AnalysisTab()
        self.audio_tab = AudioTab()
        
        tab_widget.addTab(self.record_tab, "Record")
        tab_widget.addTab(self.analysis_tab, "Analysis")
        tab_widget.addTab(self.audio_tab, "Audio")
        
        self.setCentralWidget(tab_widget)
    
    def closeEvent(self, event):
        """프로그램 종료 시 리소스 정리"""
        if hasattr(self.record_tab, 'serial_manager'):
            self.record_tab.serial_manager.close()
        if hasattr(self.record_tab, 'camera_manager'):
            self.record_tab.camera_manager.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 폰트 설정
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # 메인 윈도우 생성
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
