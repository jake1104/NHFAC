import numpy as np

try:
    from nhfac_core import PyTransform

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    from nhfac.gpu.accelerator import GPUAccelerator

    GPU_ENABLED = GPUAccelerator.is_available()
except ImportError:
    GPU_ENABLED = False


class Transforms:
    _rust_transform = PyTransform() if RUST_AVAILABLE else None

    @staticmethod
    def fft_forward(signal):
        """Standard RFFT Forward for high fidelity"""
        # 1. Try Rust
        if RUST_AVAILABLE:
            try:
                re, im = Transforms._rust_transform.fft_forward(
                    signal.astype(np.float32)
                )
                cutoff = len(signal) // 2 + 1
                return re[:cutoff] + 1j * im[:cutoff]
            except Exception:
                pass

        # 2. Try GPU
        if GPU_ENABLED:
            return GPUAccelerator.fft_forward(signal)

        # 3. Fallback to NumPy
        return np.fft.rfft(signal)

    @staticmethod
    def fft_backward(spectrum, n):
        """Standard IRFFT Backward"""
        if GPU_ENABLED:
            return GPUAccelerator.fft_backward(spectrum, n)
        return np.fft.irfft(spectrum, n=n)

    @staticmethod
    def hartley_forward(signal):
        """
        Discrete Hartley Transform (DHT) - Real-to-Real
        H(k) = sum_{n=0}^{N-1} x(n) [cos(2pi kn/N) + sin(2pi kn/N)]
        """
        # 1. Try Rust
        if RUST_AVAILABLE:
            try:
                return Transforms._rust_transform.hartley_fast(
                    signal.astype(np.float32)
                )
            except Exception:
                pass

        # 2. Try GPU
        if GPU_ENABLED:
            return GPUAccelerator.hartley_forward(signal)

        # 3. Consistent DHT identity: H = Re(FFT) - Im(FFT)
        f_val = np.fft.fft(signal)
        return f_val.real - f_val.imag

    @staticmethod
    def hartley_backward(h_coeffs):
        """
        Inverse DHT is self-inverse with 1/N scaling.
        Ensures exact reconstruction of real signals.
        """
        if GPU_ENABLED:
            return GPUAccelerator.hartley_backward(h_coeffs)

        n = len(h_coeffs)
        # Identity: DHT(DHT(x)) = N * x
        return Transforms.hartley_forward(h_coeffs) / n
