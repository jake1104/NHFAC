import numpy as np
from ..core.transforms import Transforms
from ..core.regression import AdaptiveRegression
import zlib
from .bitstream import NHFACBitstream


class NHFACDecoder:
    def __init__(self, sr=48000, frame_size=1024):
        self.sr = sr
        self.frame_size = frame_size

    def decode(self, encoded_data):
        # 0. Handle binary input
        if isinstance(encoded_data, bytes):
            encoded_data = NHFACBitstream.unpack(encoded_data)

        # 1. [Decompress & Unpack]
        h_bytes = zlib.decompress(encoded_data["h_stream"])
        t_bytes = zlib.decompress(encoded_data["t_stream"])

        if "h_shape" in encoded_data:
            h_shape = encoded_data["h_shape"]
        else:
            h_shape = (-1, self.frame_size)

        h_quant = np.frombuffer(h_bytes, dtype=np.int16).reshape(h_shape)

        # 2. handle infered shape from bitstream
        if h_quant.shape[0] == -1:
            n_rows = len(h_bytes) // (h_shape[1] * 2)
            h_quant = h_quant[:n_rows].reshape(n_rows, h_shape[1])

        # 3. Handle thetas
        n_harm = encoded_data.get("n_harmonics", 4)
        degree = encoded_data.get("degree", 0)
        n_theta_coeffs = (degree + 1) + (2 * n_harm)
        thetas = np.frombuffer(t_bytes, dtype=np.float32).reshape((-1, n_theta_coeffs))

        q_scale = encoded_data.get("q_scale", 1.0)
        factor = encoded_data.get("factor", 1e4)
        orig_len = encoded_data["orig_len"]
        pad_len = encoded_data.get("pad_len", 0)

        # 4. [Inverse Quantization]
        h_delta = h_quant.astype(np.float32) / q_scale

        # 5. [Inverse Delta Encoding]
        h_log = np.cumsum(h_delta, axis=0)

        # 6. [Inverse Log Scale]
        h_matrix = np.sign(h_log) * (np.expm1(np.abs(h_log)) / factor)

        # 7. [Signal Reconstruction (OLA)]
        step = self.frame_size // 2
        n_frames = h_matrix.shape[0]
        total_len = (n_frames - 1) * step + self.frame_size

        decoded_residual = np.zeros(total_len)
        overlap_weight = np.zeros(total_len)
        window = np.hanning(self.frame_size)

        for idx in range(n_frames):
            i = idx * step
            h_coeffs = h_matrix[idx]
            frame_res = Transforms.hartley_backward(h_coeffs)

            decoded_residual[i : i + self.frame_size] += frame_res
            overlap_weight[i : i + self.frame_size] += window

        overlap_weight[overlap_weight < 1e-6] = 1.0
        decoded_residual /= overlap_weight

        # 8. [Regression Reconstruction]
        regressor = AdaptiveRegression(None, degree=degree, n_harmonics=n_harm)
        reconstructed_reg = regressor.reconstruct_from_thetas(
            thetas, total_len, window_size=self.frame_size, step=step
        )

        full_signal = decoded_residual + reconstructed_reg

        if pad_len > 0:
            return full_signal[pad_len : pad_len + orig_len]
        return full_signal[:orig_len]
