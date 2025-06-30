import sys
import os
import librosa
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt

from analysis.impact_detection import ImpactDetector
from analysis.periodicity_detection import PeriodicityDetector
from analysis.envelope_analysis import EnvelopeAnalyzer
from ui.plot_widgets import SpectrogramWidget, MfccWidget, FftWidget
from ui.analysis_widgets import ImpactAnalysisWidget, PeriodicityAnalysisWidget, EnvelopeAnalysisWidget

class SoundAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Advanced Sound Analyzer')
        self.setGeometry(100, 100, 1200, 900)

        # Main container widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Drop area
        self.drop_label = QLabel('Drag and drop a WAV file here')
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("border: 2px dashed #aaa; padding: 20px;")
        main_layout.addWidget(self.drop_label)

        # Scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QGridLayout(content_widget)

        # Basic plots (left column)
        self.spectrogram_widget = SpectrogramWidget()
        self.mfcc_widget = MfccWidget()
        self.fft_widget = FftWidget()
        content_layout.addWidget(self.spectrogram_widget, 0, 0)
        content_layout.addWidget(self.mfcc_widget, 1, 0)
        content_layout.addWidget(self.fft_widget, 2, 0)

        # Analysis sections (right column)
        self.impact_widget = ImpactAnalysisWidget()
        self.periodicity_widget = PeriodicityAnalysisWidget()
        self.envelope_widget = EnvelopeAnalysisWidget()
        content_layout.addWidget(self.impact_widget, 0, 1)
        content_layout.addWidget(self.periodicity_widget, 1, 1)
        content_layout.addWidget(self.envelope_widget, 2, 1)

        # Time slider for FFT (below plots)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.valueChanged.connect(self.update_fft_plot)
        content_layout.addWidget(self.time_slider, 3, 0, 1, 2) # Span across both columns

        # Analysis modules
        self.impact_detector = ImpactDetector()
        self.periodicity_detector = PeriodicityDetector()
        self.envelope_analyzer = EnvelopeAnalyzer()

        self.audio = None
        self.sr = None
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.wav'):
                self.load_audio(file_path)
                break

    def load_audio(self, file_path):
        self.audio, self.sr = librosa.load(file_path, sr=None)
        self.drop_label.setText(f'Loaded: {os.path.basename(file_path)}')

        # Update basic plots
        self.spectrogram_widget.plot(self.audio, self.sr)
        self.mfcc_widget.plot(self.audio, self.sr)
        self.time_slider.setMaximum(len(self.audio) - 1)
        self.update_fft_plot(0)

        # Run and display analyses
        self.run_all_analyses()

    def update_fft_plot(self, value):
        if self.audio is not None:
            window_size = 2048
            segment = self.audio[value:value + window_size]
            self.fft_widget.plot(segment, self.sr)

    def run_all_analyses(self):
        if self.audio is None:
            return

        # Impact analysis
        self.impact_detector.set_audio(self.audio, self.sr)
        impact_results = self.impact_detector.detect_all()
        self.impact_widget.update_results(self.audio, self.sr, impact_results)

        # Periodicity analysis
        self.periodicity_detector.set_audio(self.audio, self.sr)
        periodicity_results = self.periodicity_detector.detect_all()
        self.periodicity_widget.update_results(self.audio, self.sr, periodicity_results)

        # Envelope analysis
        self.envelope_analyzer.set_audio(self.audio, self.sr)
        envelope_results = self.envelope_analyzer.analyze_all()
        self.envelope_widget.update_results(self.audio, self.sr, envelope_results)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SoundAnalyzer()
    window.show()
    sys.exit(app.exec_())