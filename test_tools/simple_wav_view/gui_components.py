#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Components Module
GUI 컴포넌트들 (파일 리스트, 시각화 프레임 등)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import logging


class FileListFrame(ctk.CTkFrame):
    """파일 리스트 프레임"""
    
    def __init__(self, parent, on_file_select=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.on_file_select = on_file_select
        self.wav_files = []
        
        # 프레임 설정
        self.configure(width=300)
        
        # GUI 구성
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 구성"""
        # 제목
        title_label = ctk.CTkLabel(self, text="WAV Files", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=10)
        
        # 파일 리스트 (Treeview 사용)
        self.setup_file_tree()
        
        # 새로고침 버튼
        refresh_btn = ctk.CTkButton(
            self,
            text="새로고침",
            command=self.refresh_file_list,
            width=100
        )
        refresh_btn.pack(pady=5)
        
    def setup_file_tree(self):
        """파일 트리뷰 설정"""
        # 트리뷰 프레임
        tree_frame = ctk.CTkFrame(self)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 스크롤바와 트리뷰
        self.tree = ttk.Treeview(tree_frame, columns=('size', 'status'), show='tree headings')
        
        # 컬럼 설정
        self.tree.heading('#0', text='파일명')
        self.tree.heading('size', text='크기')
        self.tree.heading('status', text='상태')
        
        self.tree.column('#0', width=180)
        self.tree.column('size', width=60)
        self.tree.column('status', width=50)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 패킹
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 선택 이벤트 바인딩
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
    def update_file_list(self, wav_files):
        """파일 리스트 업데이트"""
        self.wav_files = wav_files
        
        # 기존 항목 삭제
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # 새 항목 추가
        for wav_file in wav_files:
            wav_path = Path(wav_file)
            h5_path = wav_path.with_suffix('.h5')
            
            # 파일 크기
            size_mb = wav_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f}MB"
            
            # 상태 (HDF5 존재 여부)
            status = "H5" if h5_path.exists() else "WAV"
            
            # 트리에 추가
            self.tree.insert('', 'end', text=wav_path.name, values=(size_str, status))
            
    def on_tree_select(self, event):
        """트리 선택 이벤트"""
        selection = self.tree.selection()
        if selection and self.on_file_select:
            item = selection[0]
            filename = self.tree.item(item, 'text')
            
            # 전체 경로 찾기
            for wav_file in self.wav_files:
                if Path(wav_file).name == filename:
                    self.on_file_select(wav_file)
                    break
                    
    def refresh_file_list(self):
        """파일 리스트 새로고침"""
        if self.wav_files:
            self.update_file_list(self.wav_files)


class AudioVisualizationFrame(ctk.CTkFrame):
    """오디오 시각화 프레임"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.current_audio_data = None
        
        # GUI 구성
        self.setup_gui()
        
    def setup_gui(self):
        """GUI 구성"""
        # 탭뷰 생성
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 탭 추가
        self.tabview.add("Waveform")
        self.tabview.add("Spectrum")
        self.tabview.add("Spectrogram")
        
        # 각 탭에 matplotlib 캔버스 추가
        self.setup_waveform_tab()
        self.setup_spectrum_tab()
        self.setup_spectrogram_tab()
        
    def setup_waveform_tab(self):
        """Waveform 탭 설정"""
        tab = self.tabview.tab("Waveform")
        
        # matplotlib figure
        self.waveform_fig = Figure(figsize=(10, 6), dpi=100)
        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, tab)
        self.waveform_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 초기 플롯
        ax = self.waveform_fig.add_subplot(111)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
    def setup_spectrum_tab(self):
        """Spectrum 탭 설정"""
        tab = self.tabview.tab("Spectrum")
        
        # 컨트롤 프레임
        control_frame = ctk.CTkFrame(tab)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # 로그 스케일 체크박스
        self.log_scale_var = ctk.BooleanVar(value=True)
        log_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Log Scale",
            variable=self.log_scale_var,
            command=self.update_spectrum_plot
        )
        log_checkbox.pack(side="left", padx=5, pady=5)
        
        # matplotlib figure
        self.spectrum_fig = Figure(figsize=(10, 6), dpi=100)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, tab)
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 초기 플롯
        ax = self.spectrum_fig.add_subplot(111)
        ax.set_title("Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        
    def setup_spectrogram_tab(self):
        """Spectrogram 탭 설정"""
        tab = self.tabview.tab("Spectrogram")
        
        # 컨트롤 프레임
        control_frame = ctk.CTkFrame(tab)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # 채널 선택
        ctk.CTkLabel(control_frame, text="Channel:").pack(side="left", padx=5)
        self.channel_var = ctk.StringVar(value="0")
        self.channel_combo = ctk.CTkComboBox(
            control_frame,
            variable=self.channel_var,
            values=["0"],
            command=self.update_spectrogram_plot,
            width=80
        )
        self.channel_combo.pack(side="left", padx=5, pady=5)
        
        # 컬러맵 선택
        ctk.CTkLabel(control_frame, text="Colormap:").pack(side="left", padx=(20, 5))
        self.colormap_var = ctk.StringVar(value="viridis")
        colormap_combo = ctk.CTkComboBox(
            control_frame,
            variable=self.colormap_var,
            values=["viridis", "plasma", "inferno", "magma", "jet", "hot"],
            command=self.update_spectrogram_plot,
            width=100
        )
        colormap_combo.pack(side="left", padx=5, pady=5)
        
        # vmin 설정
        ctk.CTkLabel(control_frame, text="vmin:").pack(side="left", padx=(20, 5))
        self.vmin_var = ctk.StringVar(value="auto")
        self.vmin_entry = ctk.CTkEntry(
            control_frame,
            textvariable=self.vmin_var,
            width=60,
            placeholder_text="auto"
        )
        self.vmin_entry.pack(side="left", padx=5, pady=5)
        self.vmin_entry.bind("<Return>", self.update_spectrogram_plot)
        
        # vmax 설정
        ctk.CTkLabel(control_frame, text="vmax:").pack(side="left", padx=(10, 5))
        self.vmax_var = ctk.StringVar(value="auto")
        self.vmax_entry = ctk.CTkEntry(
            control_frame,
            textvariable=self.vmax_var,
            width=60,
            placeholder_text="auto"
        )
        self.vmax_entry.pack(side="left", padx=5, pady=5)
        self.vmax_entry.bind("<Return>", self.update_spectrogram_plot)
        
        # 적용 버튼
        apply_btn = ctk.CTkButton(
            control_frame,
            text="적용",
            command=self.update_spectrogram_plot,
            width=60
        )
        apply_btn.pack(side="left", padx=5, pady=5)
        
        # matplotlib figure
        self.spectrogram_fig = Figure(figsize=(10, 6), dpi=100)
        self.spectrogram_canvas = FigureCanvasTkAgg(self.spectrogram_fig, tab)
        self.spectrogram_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 초기 플롯
        ax = self.spectrogram_fig.add_subplot(111)
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        
    def update_visualization(self, audio_data):
        """시각화 업데이트"""
        self.current_audio_data = audio_data
        
        # 채널 수에 따라 채널 선택 콤보박스 업데이트
        num_channels = audio_data['metadata']['num_channels']
        channel_values = [str(i) for i in range(num_channels)]
        self.channel_combo.configure(values=channel_values)
        
        # 각 탭 업데이트
        self.update_waveform_plot()
        self.update_spectrum_plot()
        self.update_spectrogram_plot()
        
    def update_waveform_plot(self):
        """Waveform 플롯 업데이트"""
        if not self.current_audio_data:
            return
            
        try:
            # 데이터 준비
            raw_data = self.current_audio_data['raw_data']
            metadata = self.current_audio_data['metadata']
            sample_rate = metadata['sampling_rate']
            
            # 시간 축 생성
            time_axis = np.arange(raw_data.shape[1]) / sample_rate
            
            # 플롯 클리어 및 업데이트
            self.waveform_fig.clear()
            
            num_channels = raw_data.shape[0]
            
            for ch in range(num_channels):
                ax = self.waveform_fig.add_subplot(num_channels, 1, ch + 1)
                ax.plot(time_axis, raw_data[ch], linewidth=0.5)
                ax.set_title(f"Channel {ch}")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.3)
                
                if ch == num_channels - 1:
                    ax.set_xlabel("Time (s)")
                    
            self.waveform_fig.tight_layout()
            self.waveform_canvas.draw()
            
        except Exception as e:
            logging.error(f"Waveform 플롯 업데이트 실패: {str(e)}")
            
    def update_spectrum_plot(self, *args):
        """Spectrum 플롯 업데이트"""
        if not self.current_audio_data:
            return
            
        try:
            # 데이터 준비
            spectrum_data = self.current_audio_data['spectrum_data']
            metadata = self.current_audio_data['metadata']
            sample_rate = metadata['sampling_rate']
            
            # 주파수 축 생성
            freq_bins = spectrum_data.shape[1]
            freq_axis = np.linspace(0, sample_rate / 2, freq_bins)
            
            # 플롯 클리어 및 업데이트
            self.spectrum_fig.clear()
            
            num_channels = spectrum_data.shape[0]
            
            for ch in range(num_channels):
                ax = self.spectrum_fig.add_subplot(num_channels, 1, ch + 1)
                
                magnitude = spectrum_data[ch]
                
                if self.log_scale_var.get():
                    # 로그 스케일
                    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # 0 방지
                    ax.plot(freq_axis, magnitude_db)
                    ax.set_ylabel("Magnitude (dB)")
                else:
                    # 선형 스케일
                    ax.plot(freq_axis, magnitude)
                    ax.set_ylabel("Magnitude")
                    
                ax.set_title(f"Spectrum - Channel {ch}")
                ax.grid(True, alpha=0.3)
                
                if ch == num_channels - 1:
                    ax.set_xlabel("Frequency (Hz)")
                    
            self.spectrum_fig.tight_layout()
            self.spectrum_canvas.draw()
            
        except Exception as e:
            logging.error(f"Spectrum 플롯 업데이트 실패: {str(e)}")
            
    def update_spectrogram_plot(self, *args):
        """Spectrogram 플롯 업데이트"""
        if not self.current_audio_data:
            return
            
        try:
            # 선택된 채널
            channel = int(self.channel_var.get())
            
            # 데이터 준비
            stft_data = self.current_audio_data['stft_data']
            metadata = self.current_audio_data['metadata']
            sample_rate = metadata['sampling_rate']
            
            if channel >= stft_data.shape[0]:
                return
                
            # STFT 데이터 (magnitude)
            stft_magnitude = stft_data[channel]  # [frames, freq_bins]
            
            # 시간 및 주파수 축
            num_frames, num_freq_bins = stft_magnitude.shape
            time_axis = np.arange(num_frames) * (metadata['num_samples'] / sample_rate) / num_frames
            freq_axis = np.linspace(0, sample_rate / 2, num_freq_bins)
            
            # 플롯 클리어 및 업데이트
            self.spectrogram_fig.clear()
            ax = self.spectrogram_fig.add_subplot(111)
            
            # 로그 스케일 변환
            stft_db = 20 * np.log10(stft_magnitude.T + 1e-10)  # transpose for correct orientation
            
            # vmin, vmax 값 처리
            vmin_val = None
            vmax_val = None
            
            try:
                vmin_text = self.vmin_var.get().strip()
                if vmin_text and vmin_text.lower() != "auto":
                    vmin_val = float(vmin_text)
            except ValueError:
                pass  # auto 또는 잘못된 값인 경우 None 유지
                
            try:
                vmax_text = self.vmax_var.get().strip()
                if vmax_text and vmax_text.lower() != "auto":
                    vmax_val = float(vmax_text)
            except ValueError:
                pass  # auto 또는 잘못된 값인 경우 None 유지
            
            # 이미지 플롯
            im = ax.imshow(
                stft_db,
                aspect='auto',
                origin='lower',
                extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
                cmap=self.colormap_var.get(),
                vmin=vmin_val,
                vmax=vmax_val
            )
            
            ax.set_title(f"Spectrogram - Channel {channel}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            
            # 컬러바 추가
            cbar = self.spectrogram_fig.colorbar(im, ax=ax)
            cbar.set_label("Magnitude (dB)")
            
            self.spectrogram_fig.tight_layout()
            self.spectrogram_canvas.draw()
            
        except Exception as e:
            logging.error(f"Spectrogram 플롯 업데이트 실패: {str(e)}")
            
    def clear_all_plots(self):
        """모든 플롯 클리어"""
        self.waveform_fig.clear()
        self.spectrum_fig.clear()
        self.spectrogram_fig.clear()
        
        self.waveform_canvas.draw()
        self.spectrum_canvas.draw()
        self.spectrogram_canvas.draw()
        
        self.current_audio_data = None