import numpy as np
import librosa
from scipy.signal import hilbert

class EnvelopeAnalyzer:
    def __init__(self):
        self.audio = None
        self.sr = None

    def set_audio(self, audio, sr):
        self.audio = audio
        self.sr = sr

    def analyze_hilbert(self):
        """Analyze envelope using Hilbert transform."""
        analytic_signal = hilbert(self.audio)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def analyze_rms(self):
        """Analyze envelope using RMS energy."""
        frame_length = 2048
        hop_length = 512
        rms_envelope = librosa.feature.rms(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        return rms_envelope

    def analyze_onset_strength(self):
        """Analyze envelope using onset strength."""
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
        return onset_env

    def get_dominant_frequency(self, envelope):
        """Calculate the dominant frequency of an envelope."""
        if len(envelope) < 2:
            return 0
        fft_env = np.abs(np.fft.rfft(envelope))
        # The time difference between envelope points depends on the method
        # For simplicity, we'll assume a hop length of 512 for RMS and onset strength
        if len(envelope) == len(librosa.onset.onset_strength(y=self.audio, sr=self.sr)):
             hop_length = 512
        elif len(envelope) == len(librosa.feature.rms(y=self.audio, frame_length=2048, hop_length=512)[0]):
            hop_length = 512
        else: # Hilbert
            hop_length = 1

        freqs = np.fft.rfftfreq(len(envelope), hop_length/self.sr)
        dom_freq = freqs[np.argmax(fft_env[1:])+1] if len(fft_env) > 1 else 0
        return dom_freq

    def analyze_all(self):
        """Run all envelope analysis methods and get dominant frequencies."""
        hilbert_env = self.analyze_hilbert()
        rms_env = self.analyze_rms()
        onset_env = self.analyze_onset_strength()

        results = {
            'hilbert': {
                'envelope': hilbert_env,
                'dominant_frequency': self.get_dominant_frequency(hilbert_env)
            },
            'rms': {
                'envelope': rms_env,
                'dominant_frequency': self.get_dominant_frequency(rms_env)
            },
            'onset_strength': {
                'envelope': onset_env,
                'dominant_frequency': self.get_dominant_frequency(onset_env)
            }
        }
        return results