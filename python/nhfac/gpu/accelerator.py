import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class GPUAccelerator:
    """
    NHFAC GPU 가속기
    CuPy를 사용하여 Hartley 변환 및 FFT 연산을 GPU에서 수행합니다.
    """

    @staticmethod
    def is_available():
        if not GPU_AVAILABLE:
            return False
        try:
            # 실질적으로 GPU 사용 가능한지 체크
            cp.cuda.Device(0).use()
            return True
        except Exception:
            return False

    @staticmethod
    def to_gpu(array):
        """배열을 GPU로 이동"""
        if not GPU_AVAILABLE:
            return array
        return cp.asarray(array)

    @staticmethod
    def to_cpu(array):
        """배열을 CPU로 이동"""
        if not GPU_AVAILABLE:
            return array
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array

    @staticmethod
    def hartley_forward(signal):
        """
        GPU 기반 Discrete Hartley Transform (DHT)
        H = Re(FFT) - Im(FFT)
        """
        if not GPU_AVAILABLE:
            # Fallback to CPU-based logic (but this should be handled by the caller)
            f_val = np.fft.fft(signal)
            return f_val.real - f_val.imag

        sig_gpu = cp.asarray(signal)
        spec_gpu = cp.fft.fft(sig_gpu)
        hartley_gpu = spec_gpu.real - spec_gpu.imag
        return cp.asnumpy(hartley_gpu)

    @staticmethod
    def hartley_backward(h_coeffs):
        """
        GPU 기반 Inverse DHT
        """
        n = len(h_coeffs)
        if not GPU_AVAILABLE:
            return GPUAccelerator.hartley_forward(h_coeffs) / n

        h_gpu = cp.asarray(h_coeffs)
        spec_gpu = cp.fft.fft(h_gpu)
        # DHT is self-inverse with 1/N scaling
        res_gpu = (spec_gpu.real - spec_gpu.imag) / n
        return cp.asnumpy(res_gpu)

    @staticmethod
    def fft_forward(signal):
        """GPU 기반 RFFT"""
        if not GPU_AVAILABLE:
            return np.fft.rfft(signal)

        sig_gpu = cp.asarray(signal)
        spec_gpu = cp.fft.rfft(sig_gpu)
        return cp.asnumpy(spec_gpu)

    @staticmethod
    def fft_backward(spectrum, n):
        """GPU 기반 IRFFT"""
        if not GPU_AVAILABLE:
            return np.fft.irfft(spectrum, n=n)

        spec_gpu = cp.asarray(spectrum)
        res_gpu = cp.fft.irfft(spec_gpu, n=n)
        return cp.asnumpy(res_gpu)
