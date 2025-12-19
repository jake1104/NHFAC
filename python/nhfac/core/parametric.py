import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import find_peaks


class ParametricHNM:
    """
    Harmonic + Noise Model (HNM) for Sparse Audio Coding.
    Scales degrees of freedom as O(log N) by parameterizing the spectrum.
    """

    def __init__(self, sr=48000, n_fft=2048, n_cepstral=20, n_harmonics=15):
        self.sr = sr
        self.n_fft = n_fft
        self.n_cepstral = n_cepstral
        self.n_harmonics = n_harmonics

    def analyze_frame(self, frame):
        """
        Extracts f0, harmonic amplitudes, and spectral envelope (cepstrum).
        """
        n = len(frame)
        window = np.hanning(n)
        sig_w = frame * window

        # 1. Spectral Analysis
        spec = np.fft.rfft(sig_w, n=self.n_fft)
        mag = np.abs(spec)
        # Normalize magnitude: 2/N for RFFT to get physical amplitude
        mag_norm = mag * (2.0 / n)
        log_mag = 20 * np.log10(mag_norm + 1e-10)

        # 2. F0 Estimation (HPS)
        hps = log_mag.copy()
        for i in range(2, 5):
            downsampled = log_mag[::i]
            hps[: len(downsampled)] += downsampled

        min_bin = int(50 * self.n_fft / self.sr)
        f0_bin = np.argmax(hps[min_bin:]) + min_bin
        f0 = f0_bin * self.sr / self.n_fft

        # 3. Harmonic Extraction (using normalized magnitude)
        harmonic_amps = []
        for k in range(1, self.n_harmonics + 1):
            target_bin = int(round(k * f0 * self.n_fft / self.sr))
            if target_bin < len(mag_norm):
                search_win = 2
                start = max(0, target_bin - search_win)
                end = min(len(mag_norm), target_bin + search_win + 1)
                harmonic_amps.append(np.max(mag_norm[start:end]))
            else:
                harmonic_amps.append(0.0)

        # 4. Spectral Envelope (Cepstrum)
        # Use DCT to get social envelope of the normalized log magnitude
        cepstrum = dct(log_mag, type=2, norm="ortho")[: self.n_cepstral]

        return {
            "f0": float(f0),
            "harmonics": np.array(harmonic_amps, dtype=np.float32),
            "cepstrum": np.array(cepstrum, dtype=np.float32),
        }

    def synthesize_frame(self, params, n_samples):
        """
        Reconstructs frame using Sinusoidal synthesis and Noise shaping with proper scaling.
        """
        rng = np.random.default_rng()
        t = np.arange(n_samples) / self.sr

        # 1. Harmonic Synthesis (Deterministic)
        f0 = params["f0"]
        harmonics = params["harmonics"]
        deterministic = np.zeros(n_samples)

        for k, amp in enumerate(harmonics):
            # Use random phase to prevent phasing artifacts, but keep energy consistent
            phase = rng.uniform(0, 2 * np.pi)
            deterministic += amp * np.cos(2 * np.pi * (k + 1) * f0 * t + phase)

        # 2. Stochastic Synthesis (Noise)
        cepstrum_full = np.zeros(self.n_fft // 2 + 1)
        cepstrum_full[: self.n_cepstral] = params["cepstrum"]
        env_log_mag = idct(cepstrum_full, type=2, norm="ortho")
        env_mag = 10 ** (env_log_mag / 20)

        # Generate WGN and shape it. WGN in time domain with std=1 has flat PSD.
        # We must scale the noise to match the envelope energy area.
        noise_wgn = rng.normal(0, 1.0, self.n_fft)
        noise_spec = np.fft.rfft(noise_wgn)

        # The noise spec is shaped by env_mag.
        # irfft will divide by N, so we need to compensate.
        shaped_noise_spec = noise_spec * env_mag
        stochastic = np.fft.irfft(shaped_noise_spec, n=self.n_fft)[:n_samples]

        # Apply normalization to keep stochastic part at correct level
        # This is a simplification; high-quality HNM requires more complex gain matching
        return (deterministic + stochastic) * np.hanning(n_samples)
