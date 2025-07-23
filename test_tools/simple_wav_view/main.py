#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analysis & Preprocessing Tool
메인 GUI 애플리케이션
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
from pathlib import Path

# 로컬 모듈 import
from audio_processor import AudioProcessor
from hdf5_manager import HDF5Manager
from gui_components import AudioVisualizationFrame, FileListFrame
from feature_calculator import FeatureCalculator


class AudioAnalysisApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        # customtkinter 테마 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 메인 윈도우 생성
        self.root = ctk.CTk()
        self.root.title("Audio Analysis & Preprocessing Tool")
        self.root.geometry("1400x900")
        
        # 컴포넌트 초기화
        self.audio_processor = AudioProcessor()
        self.hdf5_manager = HDF5Manager()
        self.feature_calculator = FeatureCalculator()
        
        # 상태 변수
        self.current_folder = None
        self.current_h5_file = None
        self.wav_files = []
        self.selected_file = None
        self.is_h5_mode = False  # H5 파일 모드인지 폴더 모드인지 구분
        
        # GUI 구성
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 레이아웃 설정"""
        # 메인 프레임 구성
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 상단 컨트롤 패널
        self.setup_control_panel()
        
        # 중앙 컨텐츠 영역
        self.setup_content_area()
        
        # 하단 상태바
        self.setup_status_bar()
        
    def setup_control_panel(self):
        """상단 컨트롤 패널 설정"""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # 폴더 선택 버튼
        self.folder_btn = ctk.CTkButton(
            control_frame,
            text="폴더 선택",
            command=self.select_folder,
            width=120
        )
        self.folder_btn.pack(side="left", padx=5, pady=5)
        
        # H5 파일 선택 버튼
        self.h5_btn = ctk.CTkButton(
            control_frame,
            text="H5 파일 선택",
            command=self.select_h5_file,
            width=120
        )
        self.h5_btn.pack(side="left", padx=5, pady=5)
        
        # 전처리 버튼
        self.preprocess_btn = ctk.CTkButton(
            control_frame,
            text="전처리 (WAV → HDF5)",
            command=self.start_preprocessing,
            width=150,
            state="disabled"
        )
        self.preprocess_btn.pack(side="left", padx=5, pady=5)
        
        # Feature 계산 버튼
        self.feature_btn = ctk.CTkButton(
            control_frame,
            text="Cal Features",
            command=self.calculate_features,
            width=120,
            state="disabled"
        )
        self.feature_btn.pack(side="left", padx=5, pady=5)
        
        # 선택된 경로 표시
        self.folder_label = ctk.CTkLabel(
            control_frame,
            text="폴더 또는 H5 파일이 선택되지 않았습니다.",
            anchor="w"
        )
        self.folder_label.pack(side="left", padx=20, pady=5, fill="x", expand=True)
        
    def setup_content_area(self):
        """중앙 컨텐츠 영역 설정"""
        content_frame = ctk.CTkFrame(self.main_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 좌측: 파일 리스트
        self.file_list_frame = FileListFrame(
            content_frame,
            on_file_select=self.on_file_selected
        )
        self.file_list_frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # 우측: 시각화 영역
        self.visualization_frame = AudioVisualizationFrame(content_frame)
        self.visualization_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
    def setup_status_bar(self):
        """하단 상태바 설정"""
        self.status_bar = ctk.CTkLabel(
            self.main_frame,
            text="준비됨",
            anchor="w"
        )
        self.status_bar.pack(fill="x", padx=5, pady=2)
        
    def select_folder(self):
        """폴더 선택 및 WAV 파일 스캔"""
        folder_path = filedialog.askdirectory(title="WAV 파일이 있는 폴더를 선택하세요")
        
        if folder_path:
            self.current_folder = Path(folder_path)
            self.current_h5_file = None
            self.is_h5_mode = False
            self.folder_label.configure(text=f"선택된 폴더: {folder_path}")
            
            # WAV 파일 스캔
            self.scan_wav_files()
            
            # 버튼 활성화
            self.preprocess_btn.configure(state="normal")
            if self.wav_files:
                self.feature_btn.configure(state="normal")
                
    def select_h5_file(self):
        """H5 파일 선택 및 내부 WAV 파일 목록 추출"""
        h5_file_path = filedialog.askopenfilename(
            title="H5 파일을 선택하세요",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if h5_file_path:
            self.current_h5_file = Path(h5_file_path)
            self.current_folder = self.current_h5_file.parent
            self.is_h5_mode = True
            self.folder_label.configure(text=f"선택된 H5 파일: {h5_file_path}")
            
            # H5 파일에서 WAV 파일 목록 추출
            self.scan_h5_files()
            
            # 버튼 활성화 (전처리는 비활성화, Feature 계산만 활성화)
            self.preprocess_btn.configure(state="disabled")
            if self.wav_files:
                self.feature_btn.configure(state="normal")
                
    def scan_wav_files(self):
        """폴더에서 WAV 파일 스캔"""
        try:
            self.wav_files = list(self.current_folder.glob("*.wav"))
            self.wav_files.extend(list(self.current_folder.glob("*.WAV")))
            
            # 파일 리스트 업데이트
            self.file_list_frame.update_file_list(self.wav_files)
            
            self.status_bar.configure(text=f"WAV 파일 {len(self.wav_files)}개 발견")
            
        except Exception as e:
            messagebox.showerror("오류", f"파일 스캔 중 오류 발생: {str(e)}")
            
    def scan_h5_files(self):
        """H5 파일에서 WAV 파일 목록 추출"""
        try:
            import h5py
            
            wav_file_names = []
            with h5py.File(self.current_h5_file, 'r') as h5f:
                # H5 파일 내의 그룹 이름들을 WAV 파일명으로 사용
                for group_name in h5f.keys():
                    # 그룹명에 .wav 확장자 추가하여 가상 WAV 파일 경로 생성
                    wav_file_path = self.current_folder / f"{group_name}.wav"
                    wav_file_names.append(wav_file_path)
            
            self.wav_files = wav_file_names
            
            # 파일 리스트 업데이트
            self.file_list_frame.update_file_list(self.wav_files)
            
            self.status_bar.configure(text=f"H5 파일에서 {len(self.wav_files)}개 WAV 데이터 발견")
            
        except Exception as e:
            messagebox.showerror("오류", f"H5 파일 스캔 중 오류 발생: {str(e)}")
            
    def start_preprocessing(self):
        """전처리 시작 (별도 스레드에서 실행)"""
        if self.is_h5_mode:
            messagebox.showinfo("정보", "H5 파일 모드에서는 전처리가 필요하지 않습니다.")
            return
            
        if not self.wav_files:
            messagebox.showwarning("경고", "처리할 WAV 파일이 없습니다.")
            return
            
        # 전처리 확인
        result = messagebox.askyesno(
            "전처리 확인",
            f"{len(self.wav_files)}개의 WAV 파일을 HDF5로 변환하시겠습니까?\n"
            "이 작업은 시간이 걸릴 수 있습니다."
        )
        
        if result:
            # 버튼 비활성화
            self.preprocess_btn.configure(state="disabled", text="전처리 중...")
            
            # 별도 스레드에서 전처리 실행
            thread = threading.Thread(target=self.run_preprocessing)
            thread.daemon = True
            thread.start()
            
    def run_preprocessing(self):
        """전처리 실행 (백그라운드)"""
        try:
            # 새로운 구조: 일괄 변환 사용
            self.root.after(0, lambda: 
                self.status_bar.configure(text="전처리 중... (일괄 변환)"))
            
            # HDF5Manager의 일괄 변환 기능 사용
            results = self.hdf5_manager.convert_wav_to_hdf5_batch(
                self.wav_files, 
                self.audio_processor,
                progress_callback=lambda current, total, msg: 
                    self.root.after(0, lambda: 
                        self.status_bar.configure(text=f"전처리 중... ({current}/{total}) {msg}"))
            )
            
            # 완료 처리
            self.root.after(0, self.preprocessing_completed)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("오류", f"전처리 중 오류 발생: {str(e)}"))
            self.root.after(0, self.preprocessing_completed)
            
    def preprocessing_completed(self):
        """전처리 완료 처리"""
        self.preprocess_btn.configure(state="normal", text="전처리 (WAV → HDF5)")
        self.status_bar.configure(text="전처리 완료")
        messagebox.showinfo("완료", "전처리가 완료되었습니다.")
        
        # 파일 리스트 새로고침
        self.file_list_frame.refresh_file_list()
        
    def on_file_selected(self, file_path):
        """파일 선택 시 호출"""
        self.selected_file = Path(file_path)
        
        # 파일 로드 및 시각화
        self.load_and_visualize_file()
        
    def load_and_visualize_file(self):
        """선택된 파일 로드 및 시각화"""
        if not self.selected_file:
            return
            
        try:
            # 새로운 HDF5 구조에서 로드 시도 (폴더명.h5 파일에서 WAV 파일명 그룹)
            audio_data = self.hdf5_manager.load_audio_data(self.selected_file)
            
            if not audio_data:
                # HDF5에서 로드 실패시 WAV에서 직접 로드
                audio_data = self.audio_processor.load_wav(self.selected_file)
                if audio_data:
                    # 실시간 STFT/Spectrum 계산
                    audio_data['stft_data'] = self.audio_processor.compute_stft(audio_data['raw_data'])
                    audio_data['spectrum_data'] = self.audio_processor.compute_spectrum(audio_data['raw_data'])
                    
            if audio_data:
                # 시각화 업데이트
                self.visualization_frame.update_visualization(audio_data)
                self.status_bar.configure(text=f"로드됨: {self.selected_file.name}")
            else:
                messagebox.showerror("오류", "파일을 로드할 수 없습니다.")
                
        except Exception as e:
            messagebox.showerror("오류", f"파일 로드 중 오류 발생: {str(e)}")
            
    def calculate_features(self):
        """Feature 계산 및 히스토그램 표시"""
        if not self.wav_files:
            messagebox.showwarning("경고", "계산할 파일이 없습니다.")
            return
            
        try:
            # Feature 계산
            features = self.feature_calculator.calculate_features_for_files(self.wav_files)
            
            if features:
                # 히스토그램 표시
                self.feature_calculator.show_histogram(features)
                self.status_bar.configure(text=f"Feature 계산 완료: {len(features)}개 파일")
            else:
                messagebox.showwarning("경고", "계산된 Feature가 없습니다.")
                
        except Exception as e:
            messagebox.showerror("오류", f"Feature 계산 중 오류 발생: {str(e)}")
            
    def run(self):
        """애플리케이션 실행"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AudioAnalysisApp()
    app.run()