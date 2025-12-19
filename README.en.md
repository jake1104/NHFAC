# NHFAC (Nonlinear Hartley-Fourier Audio Codec) - v0.1.0

[English](./README.en.md) | [한국어](./README.md)

NHFAC (Nonlinear Hartley-Fourier Audio Codec) is a next-generation high-performance audio codec that combines **custom-developed nonlinear regression algorithms** with a **proprietary Hartley-Fourier transform engine**. This project aims to provide high-quality audio compression and precise signal analysis simultaneously, pushing the boundaries of traditional methods.

## Key Features

- **Custom Adaptive Regression**: A proprietary algorithm that mathematically decomposes the structural flow and major harmonics of an audio signal (Polynomial + Sinusoidal). This maximizes compression efficiency while perfectly preserving the "spine" of the sound.
- **Proprietary Hartley-Fourier Transform**: Instead of the traditional Fast Fourier Transform (FFT) which requires complex number operations, this project utilizes a custom transform technology based on the Hartley Transform. Reimagined through a Fourier analysis lens, it operates entirely in the real domain, increasing calculation speeds and reducing memory overhead.
- **Bark-scale Psychoacoustic Model**: Achieves high-efficiency compression by applying Bark-scale thresholds, considering human auditory characteristics (Simultaneous Masking).
- **Custom Binary Bitstream**: Improved storage efficiency and security by removing `pickle` dependencies and introducing a dedicated binary format (`NHFC`).
- **Real-time Streaming**: Supports real-time audio capture, encoding, decoding, and loopback streaming via `sounddevice`.
- **Hybrid Engine**: The computation-intensive core engine is implemented in Rust (`nhfac_core`) for peak performance.
- **GPU Acceleration**: Dramatic performance boost for large-scale data via CuPy-based GPU parallel processing.
- **Visual Analysis Tool**: Real-time verification of compression processes, residual analysis, and spectral changes through `gui_analyzer.py`.

## System Requirements

- Python 3.12 or higher
- Rust (If building core extensions from source)
- CUDA-supported GPU (For GPU acceleration mode)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jake1104/NHFAC.git
   cd NHFAC
   ```

2. Dependency installation and environment setup:
   `uv` automatically handles virtual environment creation and dependency installation.

   ```bash
   uv sync
   ```

3. Build Rust Extension (Optional):

   ```bash
   uv run maturin develop -m rust/Cargo.toml
   ```

## Usage

### Running the GUI Analyzer

```bash
python gui_analyzer.py
```

Detailed analysis of the NHFAC encoding/decoding process can be performed by loading audio files through the GUI.

### Python API Example

```python
from nhfac.codec.encoder import NHFACEncoder
from nhfac.codec.decoder import NHFACDecoder
from nhfac.io.soundfile_io import AudioIO

# Load Audio
signal, sr = AudioIO.read("audio.wav")

# Encoding
encoder = NHFACEncoder(sr=sr)
encoded_data = encoder.encode(signal)

# Decoding
decoder = NHFACDecoder(sr=sr)
reconstructed = decoder.decode(encoded_data)
```

## Performance Metrics

NHFAC provides the following quality metrics:

- **Global SNR**: Total Signal-to-Noise Ratio
- **Segmental SNR**: Frame-by-frame Signal-to-Noise Ratio (SSNR)
- **LSD (Log-Spectral Distance)**: Spectral envelope fidelity error
- **Residual SNR**: Quality metric for the residual signal after **Custom Adaptive Regression**

## License

This project is licensed under the [MIT License](./LICENSE).
