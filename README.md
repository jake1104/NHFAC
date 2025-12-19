# NHFAC (Nonlinear Hartley-Fourier Audio Codec) - v0.1.0

[English](./README.en.md) | [한국어](./README.md)

NHFAC(Nonlinear Hartley-Fourier Audio Codec)는 **자체 개발한 비선형 회귀 알고리즘**과 **독자적인 하틀리-푸리에 변환 엔진**을 결합한 혁신적인 오디오 코덱입니다. 이 프로젝트는 기존 방식의 한계를 넘어 고품질 오디오 압축과 정밀한 신호 분석 기능을 동시에 제공하는 것을 목표로 합니다.

## 주요 특징

- **자체 개발 비선형 적응형 회귀 (Custom Adaptive Regression)**: 오디오 신호의 거시적 흐름과 주요 배음을 수학적으로 정교하게 먼저 분리해내는 독자 알고리즘입니다. 이를 통해 압축 효율을 극대화하고 소리의 뼈대를 완벽하게 보존합니다.
- **독자적 하틀리-푸리에 변환 (Proprietary Hartley-Fourier Transform)**: 복잡한 복소수 연산이 필요한 기존 푸리에 변환(FFT) 대신, 실수 영역 연산에 최적화된 하틀리 변환을 푸리에 분석 관점에서 재구성한 프로젝트 고유의 변환 기술입니다. 연산 속도는 높이고 메모리 낭비는 줄였습니다.
- **Bark-scale 심리음향 모델**: 인간의 청각 특성(Simultaneous Masking)을 고려한 Bark 스케일 임계치 적용으로 고효율 압축을 실현합니다.
- **커스텀 바이너리 비스트림**: `pickle` 의존성을 제거하고 전용 바이너리 포맷(`NHFC`)을 도입하여 저장 공간 및 보안성을 강화했습니다.
- **실시간 스트리밍**: `sounddevice`를 활용한 실시간 오디오 캡처, 인코딩, 디코딩 루프백 스트리밍을 지원합니다.
- **하이브리드 엔진**: 연산 집약적인 코어 엔진은 Rust(`nhfac_core`)로 구현되어 최상의 성능을 제공합니다.
- **GPU 가속**: CuPy 기반의 GPU 병렬 처리를 지원하여 대규모 데이터 처리 속도를 획기적으로 향상시켰습니다.
- **시각적 분석 도구**: `gui_analyzer.py`를 통해 신호의 압축 과정, 잔차 분석, 스펙트럼 변화를 실시간으로 확인할 수 있습니다.

## 시스템 요구 사항

- Python 3.12 이상
- Rust (코어 확장을 빌드할 경우)
- CUDA 지원 GPU (GPU 가속 모드를 사용할 경우)

## 설치 방법

1. 저장소 복제:

```bash
git clone https://github.com/jake1104/NHFAC.git
cd NHFAC
```

2. 의존성 설치 및 환경 구축:

`uv`는 가상환경 생성과 의존성 설치를 자동으로 처리합니다.

```bash
uv sync
```

_이 명령은 `.venv`를 자동으로 생성하고, `pyproject.toml`에 정의된 모든 패키지를 설치합니다._

3. Rust 확장 빌드 (선택 사항):

```bash
uv run maturin develop -m rust/Cargo.toml
```

## 사용 방법

### GUI 분석기 실행

```bash
python gui_analyzer.py
```

GUI를 통해 오디오 파일을 로드하고 NHFAC 인코딩/디코딩 과정을 상세히 분석할 수 있습니다.

### Python API 사용 예시

```python
from nhfac.codec.encoder import NHFACEncoder
from nhfac.codec.decoder import NHFACDecoder
from nhfac.io.soundfile_io import AudioIO

# 오디오 로드
signal, sr = AudioIO.read("audio.wav")

# 인코딩
encoder = NHFACEncoder(sr=sr)
encoded_data = encoder.encode(signal)

# 디코딩
decoder = NHFACDecoder(sr=sr)
reconstructed = decoder.decode(encoded_data)
```

## 성능 지표 (Metrics)

NHFAC는 다음과 같은 품질 지표를 제공합니다:

- **Global SNR**: 전체 신호 대 잡음비
- **Segmental SNR**: 구간별 신호 대 잡음비 (SSNR)
- **LSD (Log-Spectral Distance)**: 로그 스펙트럼 거리
- **Residual SNR**: 자체 개발 적응형 회귀(Adaptive Regression) 수행 후 추출된 잔차 신호의 품질 지표

## 라이선스

이 프로젝트는 [MIT License](./LICENSE)에 따라 배포됩니다.
