import numpy as np
import librosa
from scipy.signal import find_peaks, correlate
from scipy.stats import mode

class PeriodicityDetector:
    def __init__(self):
        self.audio = None
        self.sr = None

    def set_audio(self, audio, sr):
        self.audio = audio
        self.sr = sr

    def detect_pyin(self):
        """Detect periodicity using pYIN algorithm."""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        times = librosa.times_like(f0)
        return {
            'f0': f0,
            'times': times,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs
        }

    def detect_autocorrelation(self):
        """Detect periodicity using autocorrelation."""
        corr = correlate(self.audio, self.audio, mode='full')
        corr = corr[len(corr)//2:]
        peaks, _ = find_peaks(corr, distance=int(self.sr*0.05))
        if len(peaks) > 1:
            periods = np.diff(peaks) / self.sr
            frequencies = 1 / periods
            return {
                'correlation': corr,
                'peak_periods': periods,
                'frequencies': frequencies
            }
        return None

    def detect_cepstrum(self):
        """Detect periodicity using cepstral analysis."""
        spectrum = np.fft.fft(self.audio)
        log_spectrum = np.log(np.abs(spectrum) + 1e-6)
        cepstrum = np.fft.ifft(log_spectrum).real
        # Convert quefrency to frequency (focus on typical speech/music range)
        min_period = int(self.sr / 2000)  # 2000 Hz max
        max_period = int(self.sr / 50)    # 50 Hz min
        peak_idx = min_period + np.argmax(np.abs(cepstrum[min_period:max_period]))
        if peak_idx > 0:
            fundamental_freq = self.sr / peak_idx
            return {
                'cepstrum': cepstrum,
                'fundamental_frequency': fundamental_freq
            }
        return None

    def detect_harmonic_product_spectrum(self):
        """Detect periodicity using Harmonic Product Spectrum."""
        N = len(self.audio)
        spectrum = np.abs(np.fft.fft(self.audio))
        hps = np.copy(spectrum[:N//2])
        
        for h in range(2, 6):
            decimated = spectrum[::h]  # Take every h-th sample
            hps[:len(decimated)] *= decimated[:len(hps)]
        
        peak_idx = np.argmax(hps)
        frequency = peak_idx * self.sr / N
        
        return {
            'hps': hps,
            'fundamental_frequency': frequency
        }

    def detect_all(self):
        """Run all periodicity detection methods and combine results."""
        results = {
            'pyin': self.detect_pyin(),
            'autocorrelation': self.detect_autocorrelation(),
            'cepstrum': self.detect_cepstrum(),
            'hps': self.detect_harmonic_product_spectrum()
        }
        
        # Combine fundamental frequency estimates
        f0_estimates = []
        if results['pyin'] is not None:
            f0_estimates.append(np.nanmean(results['pyin']['f0']))
        if results['autocorrelation'] is not None:
            f0_estimates.append(np.mean(results['autocorrelation']['frequencies']))
        if results['cepstrum'] is not None:
            f0_estimates.append(results['cepstrum']['fundamental_frequency'])
        if results['hps'] is not None:
            f0_estimates.append(results['hps']['fundamental_frequency'])
        
        if f0_estimates:
            results['combined_f0'] = np.median(f0_estimates)
        
        return results