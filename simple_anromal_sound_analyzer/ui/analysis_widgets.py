from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt
from .plot_widgets import BasePlotWidget
import matplotlib.pyplot as plt
import numpy as np

class AnalysisResultWidget(QWidget):
    def __init__(self, title):
        super().__init__()
        self.layout = QVBoxLayout()
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.title)

        # Create tabs for different detection methods
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def add_method_tab(self, method_name, plot_widget, description):
        tab = QWidget()
        tab_layout = QVBoxLayout()

        # Add plot widget
        tab_layout.addWidget(plot_widget)

        # Add description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        tab_layout.addWidget(desc_label)

        tab.setLayout(tab_layout)
        self.tabs.addTab(tab, method_name)

class ImpactAnalysisWidget(AnalysisResultWidget):
    def __init__(self):
        super().__init__("Impact Event Detection")
        self.method_descriptions = {
            "Onset Based": "Detects impacts using librosa's onset detection, which combines multiple features including spectral flux and phase deviation.",
            "Energy Based": "Uses short-term energy analysis to identify sudden increases in signal amplitude, effective for percussion-like sounds.",
            "Spectral Flux": "Measures frame-to-frame spectral changes to detect sudden timbral variations typical of impact sounds."
        }

    def update_results(self, audio, sr, results):
        self.tabs.clear()
        for method, times in results.items():
            plot_widget = BasePlotWidget()
            def plot_func(audio, times):
                plt.plot(np.arange(len(audio))/sr, audio)
                plt.vlines(times, -1, 1, color='r', label='Impacts')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
            plot_widget.plot_data(plot_func, audio, times)
            display_name = {
                'onset_based': 'Onset Based',
                'energy_based': 'Energy Based',
                'spectral_flux': 'Spectral Flux'
            }.get(method, method.replace('_', ' ').title())
            self.add_method_tab(display_name, plot_widget, self.method_descriptions[display_name])

class PeriodicityAnalysisWidget(AnalysisResultWidget):
    def __init__(self):
        super().__init__("Periodicity Detection")
        self.method_descriptions = {
            "Pyin": "Probabilistic YIN algorithm for robust fundamental frequency estimation, particularly effective for musical sounds.",
            "Autocorrelation": "Classical method that identifies repeating patterns by comparing the signal with delayed versions of itself.",
            "Cepstrum": "Identifies periodic components by analyzing the spectrum of the log-spectrum, useful for harmonic sounds.",
            "Harmonic Product Spectrum": "Emphasizes harmonically related frequency components to estimate fundamental frequency."
        }

    def update_results(self, audio, sr, results):
        self.tabs.clear()
        for method, data in results.items():
            if method == 'combined_f0':
                continue
            plot_widget = BasePlotWidget()
            if method == 'pyin':
                def plot_func(times, f0):
                    plt.plot(times, f0, label='F0', color='cyan', linewidth=2)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Frequency (Hz)')
                plot_widget.plot_data(plot_func, data['times'], data['f0'])
            elif method == 'autocorrelation':
                def plot_func(corr):
                    plt.plot(np.arange(len(corr))/sr, corr)
                    plt.xlabel('Lag (s)')
                    plt.ylabel('Correlation')
                plot_widget.plot_data(plot_func, data['correlation'])
            elif method == 'hps':
                def plot_func(hps):
                    freqs = np.linspace(0, sr/2, len(hps))
                    plt.plot(freqs, hps)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('HPS Magnitude')
                plot_widget.plot_data(plot_func, data['hps'])
            elif method == 'cepstrum':
                def plot_func(cepstrum):
                    quefrency = np.arange(len(cepstrum)) / sr
                    plt.plot(quefrency, cepstrum)
                    plt.xlabel('Quefrency (s)')
                    plt.ylabel('Magnitude')
                plot_widget.plot_data(plot_func, data['cepstrum'])
            
            display_name = {
                'pyin': 'Pyin',
                'autocorrelation': 'Autocorrelation',
                'cepstrum': 'Cepstrum',
                'hps': 'Harmonic Product Spectrum'
            }.get(method, method.replace('_', ' ').title())
            self.add_method_tab(display_name, plot_widget, self.method_descriptions[display_name])

class EnvelopeAnalysisWidget(AnalysisResultWidget):
    def __init__(self):
        super().__init__("Envelope Analysis")
        self.method_descriptions = {
            "Hilbert": "Uses the Hilbert transform to compute the analytical signal and extract the true amplitude envelope.",
            "RMS": "Computes the root mean square energy in short windows, providing a smoothed energy envelope.",
            "Onset Strength": "Measures the likelihood of onset events, highlighting rhythmic structure in the signal."
        }

    def update_results(self, audio, sr, results):
        self.tabs.clear()
        for method, data in results.items():
            plot_widget = BasePlotWidget()
            def plot_func(envelope, dom_freq):
                times = np.arange(len(envelope)) * (1/sr if method == 'hilbert' else 512/sr)
                plt.plot(times, envelope)
                plt.title(f'Dominant Frequency: {dom_freq:.2f} Hz')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
            plot_widget.plot_data(plot_func, data['envelope'], data['dominant_frequency'])
            display_name = {
                'hilbert': 'Hilbert',
                'rms': 'RMS',
                'onset_strength': 'Onset Strength'
            }.get(method, method.replace('_', ' ').title())
            self.add_method_tab(display_name, plot_widget, self.method_descriptions[display_name])