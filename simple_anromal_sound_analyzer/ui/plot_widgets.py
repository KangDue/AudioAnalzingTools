from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa.display

class BasePlotWidget(QWidget):
    def __init__(self, title=""):
        super().__init__()
        self.layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.plot_label = QLabel()
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.plot_label)
        self.setLayout(self.layout)
        self.temp_image_path = f"{title.replace(' ', '_').lower()}_temp.png"

    def plot_data(self, plot_function, *args, **kwargs):
        plt.figure(figsize=(8, 2))
        plot_function(*args, **kwargs)
        plt.tight_layout()
        plt.savefig(self.temp_image_path)
        plt.close()
        pixmap = QPixmap(self.temp_image_path)
        self.plot_label.setPixmap(pixmap)
        os.remove(self.temp_image_path)

class SpectrogramWidget(BasePlotWidget):
    def __init__(self):
        super().__init__("Spectrogram")

    def plot(self, audio, sr):
        def plot_func(audio, sr):
            S = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
        self.plot_data(plot_func, audio, sr)

class MfccWidget(BasePlotWidget):
    def __init__(self):
        super().__init__("MFCC")

    def plot(self, audio, sr):
        def plot_func(audio, sr):
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
        self.plot_data(plot_func, audio, sr)

class FftWidget(BasePlotWidget):
    def __init__(self):
        super().__init__("FFT")

    def plot(self, segment, sr):
        def plot_func(segment, sr):
            if len(segment) == 0:
                return
            fft = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            plt.plot(freqs, fft)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)
        self.plot_data(plot_func, segment, sr)