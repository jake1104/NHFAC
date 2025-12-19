import numpy as np
from nhfac.codec.encoder import NHFACEncoder
from nhfac.codec.decoder import NHFACDecoder
from nhfac.io.soundfile_io import AudioIO
import os


def generate_test_signal(duration=1.0, sr=48000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave (Fundamental)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add some harmonics
    signal += 0.2 * np.sin(2 * np.pi * 880 * t)
    # Add a polynomial trend (to test regression)
    signal += 0.1 * (t**2) + 0.05 * t
    return signal, sr


def main():
    sr = 48000
    signal, sr = generate_test_signal()

    # Initialize Codec
    encoder = NHFACEncoder(sr=sr)
    decoder = NHFACDecoder(sr=sr)

    print("Encoding...")
    encoded = encoder.encode(signal)

    print("Decoding...")
    reconstructed = decoder.decode(encoded)

    # Ensure lengths match for comparison
    min_len = min(len(signal), len(reconstructed))
    signal = signal[:min_len]
    reconstructed = reconstructed[:min_len]

    from nhfac.core.metrics import NHFACMetrics

    metrics = NHFACMetrics.calculate_all(signal, reconstructed, encoded["residual"], sr)

    print(f"Signal Type Detected: {encoded['signal_type']}")
    print(f"Global SNR:    {metrics['snr_global']:.2f} dB")
    print(f"Segmental SNR: {metrics['ssnr']:.2f} dB")
    print(f"Residual SNR:  {metrics['snr_residual']:.2f} dB")
    print(f"LSD Error:     {metrics['lsd']:.4f}")

    # Save results
    if not os.path.exists("output"):
        os.makedirs("output")
    AudioIO.write("output/original.wav", signal, sr)
    AudioIO.write("output/reconstructed.wav", reconstructed, sr)


if __name__ == "__main__":
    main()
