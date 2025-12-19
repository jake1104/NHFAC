# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-12-19

### Added

- **Core Engine**: Initial release of NHFAC (Nonlinear Hartley-Fourier Audio Codec).
- **Custom Adaptive Regression**: Proprietary algorithm for structural signal decomposition (Polynomial + Sinusoidal).
- **Hartley-Fourier Transform**: High-efficiency real-to-real transform engine with Rust and GPU (CuPy) support.
- **Psychoacoustic Model**: Bark-scale based simultaneous masking and ATH integration.
- **Binary Bitstream**: Custom `.nhfac` format with structured headers and zlib compression.
- **GUI Analyzer**: Comprehensive visual tool for real-time signal analysis and processing.
- **Feature Extraction API**: Export internal latent features for AI/ML training.
- **Hybrid Implementation**: Performance-critical components in Rust and high-level logic in Python.
