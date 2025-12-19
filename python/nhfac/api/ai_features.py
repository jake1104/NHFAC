import numpy as np
import zlib
from ..codec.encoder import NHFACEncoder
from ..codec.decoder import NHFACDecoder
from ..core.parametric import ParametricHNM


class NHFACFeatureExtractor:
    """
    AI-Ready Feature Extraction API for NHFAC.
    Converts audio signals into structured mathematical "latent" features
    useful for ML training, searching, and classification.
    """

    def __init__(self, sr=48000, frame_size=1024):
        self.sr = sr
        self.frame_size = frame_size
        self.encoder = NHFACEncoder(sr=sr, frame_size=frame_size)
        self.hnm = ParametricHNM(sr=sr, n_fft=frame_size)

    def extract_features(self, signal):
        """
        Extracts a comprehensive feature map of the signal.
        Returns:
            dict containing:
                - thetas: Regression coefficients (Structural trend)
                - h_coeffs: Hartley coefficients (Residual texture)
                - f0: Fundamental frequency stream
                - harmonics: Harmonic amplitudes
                - cepstrum: Spectral envelope (TIMIT-like features)
        """
        # 1. Get Hartley-based encoding (current high-fidelity)
        encoded = self.encoder.encode(signal)

        # Decompress thetas
        t_bytes = zlib.decompress(encoded["t_stream"])
        n_harm_reg = encoded["n_harmonics"]
        degree = encoded["degree"]
        n_theta_coeffs = (degree + 1) + (2 * n_harm_reg)
        thetas = np.frombuffer(t_bytes, dtype=np.float32).reshape((-1, n_theta_coeffs))

        # 2. Get Parametric features (HNM)
        # This provides the "semantic" features often used in AI
        residual = encoded["residual"]
        step = self.frame_size // 2

        f0_stream = []
        harm_stream = []
        ceps_stream = []

        for i in range(0, len(residual) - self.frame_size + 1, step):
            frame = residual[i : i + self.frame_size]
            p = self.hnm.analyze_frame(frame)
            f0_stream.append(p["f0"])
            harm_stream.append(p["harmonics"])
            ceps_stream.append(p["cepstrum"])

        return {
            "structural_thetas": thetas,  # [N_frames, N_coeffs]
            "spectral_envelope": np.array(ceps_stream),  # [N_frames, 20]
            "fundamental_freq": np.array(f0_stream),  # [N_frames]
            "harmonic_amps": np.array(harm_stream),  # [N_frames, 15]
            "signal_type": encoded["signal_type"],
            "sr": self.sr,
        }

    def export_to_numpy(self, signal, output_path):
        """Exports features as a consolidated .npz file for AI training."""
        features = self.extract_features(signal)
        np.savez(output_path, **features)
        return output_path

    @staticmethod
    def batch_process(file_paths, sr=48000):
        """Static helper for batch extraction from a list of files."""
        import librosa

        extractor = NHFACFeatureExtractor(sr=sr)
        batch_results = []
        for path in file_paths:
            sig, _ = librosa.load(path, sr=sr)
            batch_results.append(extractor.extract_features(sig))
        return batch_results
