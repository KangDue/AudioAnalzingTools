import numpy as np
import librosa
from scipy.signal import find_peaks

class ImpactDetector:
    def __init__(self):
        self.audio = None
        self.sr = None

    def set_audio(self, audio, sr):
        self.audio = audio
        self.sr = sr

    def detect_onset_based(self):
        """Detect impacts using librosa's onset detection."""
        onset_env = librosa.onset.onset_detect(y=self.audio, sr=self.sr, units='time')
        return onset_env

    def detect_energy_based(self):
        """Detect impacts using short-term energy analysis."""
        frame_length = int(0.02 * self.sr)
        hop_length = int(0.01 * self.sr)
        energy = np.array([
            np.sum(np.abs(self.audio[i:i+frame_length])**2)
            for i in range(0, len(self.audio), hop_length)
        ])
        threshold = np.mean(energy) + 2 * np.std(energy)
        impacts = np.where(energy > threshold)[0]
        return impacts * hop_length / self.sr

    def detect_spectral_flux(self):
        """Detect impacts using spectral flux."""
        S = np.abs(librosa.stft(self.audio))
        flux = np.sum(np.diff(S, axis=1), axis=0)
        peaks, _ = find_peaks(flux, distance=int(self.sr*0.05), prominence=np.std(flux))
        return librosa.frames_to_time(peaks, sr=self.sr)

    def detect_all(self):
        """Run all impact detection methods and combine results."""
        results = {
            'onset_based': self.detect_onset_based(),
            'energy_based': self.detect_energy_based(),
            'spectral_flux': self.detect_spectral_flux()
        }
        return results