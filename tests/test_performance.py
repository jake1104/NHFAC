import numpy as np
import time
from nhfac.core.transforms import Transforms
from nhfac.gpu.accelerator import GPUAccelerator


def test_transforms_accuracy():
    print("\n--- Testing Transform Accuracy ---")
    sr = 48000
    n = 4096
    t = np.linspace(0, n / sr, n, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Hartley Forward/Backward
    h_coeffs = Transforms.hartley_forward(signal)
    reconstructed = Transforms.hartley_backward(h_coeffs)

    mse = np.mean((signal - reconstructed) ** 2)
    print(f"Hartley Reconstruction MSE: {mse:.2e}")
    assert mse < 1e-10

    # FFT Forward/Backward
    spectrum = Transforms.fft_forward(signal)
    reconstructed_fft = Transforms.fft_backward(spectrum, n)

    mse_fft = np.mean((signal - reconstructed_fft) ** 2)
    print(f"FFT Reconstruction MSE: {mse_fft:.2e}")
    assert mse_fft < 1e-10


def test_gpu_vs_cpu_perf():
    if not GPUAccelerator.is_available():
        print("\nGPU not available, skipping performance test.")
        return

    print("\n--- Testing GPU vs CPU Performance ---")
    n = 2**20  # 1M samples
    signal = np.random.randn(n).astype(np.float32)

    # CPU Time
    start = time.time()
    for _ in range(10):
        _ = np.fft.fft(signal)
    cpu_time = (time.time() - start) / 10
    print(f"CPU FFT Avg Time: {cpu_time*1000:.2f} ms")

    # GPU Time
    import cupy as cp

    sig_gpu = cp.asarray(signal)
    # Warmup
    _ = cp.fft.fft(sig_gpu)
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    for _ in range(10):
        _ = cp.fft.fft(sig_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) / 10
    print(f"GPU FFT Avg Time: {gpu_time*1000:.2f} ms")

    if gpu_time < cpu_time:
        print(f"GPU is {cpu_time/gpu_time:.1f}x faster!")
    else:
        print(
            "GPU is not faster for this size (unexpected but possible for small sizes)."
        )


if __name__ == "__main__":
    try:
        test_transforms_accuracy()
    except Exception as e:
        print(f"Accuracy test failed: {e}")

    try:
        test_gpu_vs_cpu_perf()
    except Exception as e:
        print(f"Performance test failed: {e}")
