import numpy as np
import zlib
from ..core.regression import AdaptiveRegression
from ..core.transforms import Transforms
from ..psycho.psychoacoustics import apply_masking
from ..signal.classifier import SignalClassifier
from .bitstream import NHFACBitstream


class NHFACEncoder:
    def __init__(self, sr=48000, frame_size=1024):
        self.sr = sr
        self.frame_size = frame_size

    def encode(self, signal):
        """
        High-fidelity NHFAC Encoder (Hartley-based)
        """
        pad_len = self.frame_size
        padded_signal = np.pad(signal, (pad_len, pad_len), mode="reflect")

        classifier = SignalClassifier()
        signal_type = classifier.classify(signal, self.sr)

        degree = 4 if signal_type == "voice" else 3
        regressor = AdaptiveRegression(padded_signal, degree=degree, n_harmonics=4)
        thetas, residual = regressor.fit_sliding_window(
            window_size=self.frame_size, step=self.frame_size // 2, decay=0.8
        )

        step = self.frame_size // 2
        window = np.hanning(self.frame_size)
        n_frames = (len(residual) - self.frame_size) // step + 1

        # 1. Spectral Processing
        h_matrix = np.zeros((n_frames, self.frame_size), dtype=np.float32)
        for idx, i in enumerate(range(0, len(residual) - self.frame_size + 1, step)):
            frame = residual[i : i + self.frame_size] * window
            h_coeffs = Transforms.hartley_forward(frame)

            # [Psychoacoustic Masking]
            h_masked = apply_masking(h_coeffs, self.sr)
            h_matrix[idx] = h_masked

        # 2. [Log Scale Transformation]
        factor = 1e4
        h_log = np.sign(h_matrix) * np.log1p(np.abs(h_matrix) * factor)

        # 3. [Inter-frame Correlation Removal (Delta Encoding)]
        h_delta = np.zeros_like(h_log)
        h_delta[0] = h_log[0]
        h_delta[1:] = h_log[1:] - h_log[:-1]

        # 4. [Quantization]
        max_val = np.abs(h_delta).max() if n_frames > 0 else 1.0
        q_scale = 32767 / (max_val + 1e-6)
        h_quant = (h_delta * q_scale).astype(np.int16)

        # 5. [Entropy Coding]
        h_compressed = zlib.compress(h_quant.tobytes(), level=9)
        thetas_compressed = zlib.compress(np.array(thetas, dtype=np.float32).tobytes())

        return {
            "h_stream": h_compressed,
            "t_stream": thetas_compressed,
            "h_shape": h_quant.shape,
            "q_scale": q_scale,
            "factor": factor,
            "residual": residual,
            "signal_type": signal_type,
            "orig_len": len(signal),
            "pad_len": pad_len,
            "degree": degree,
            "n_harmonics": 4,
            "sr": self.sr,
        }

    def encode_to_binary(self, signal):
        data = self.encode(signal)
        return NHFACBitstream.pack(data)
