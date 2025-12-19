import librosa
import numpy as np
from scipy.signal import find_peaks


class SignalClassifier:
    @staticmethod
    def classify(signal, sr):
        """음성/악기/노이즈 분류 (Voice/Instrument/Noise Classification)"""
        # 스펙트럼 특성 추출
        D = librosa.stft(signal)
        magnitude = np.abs(D)

        # 주파수 축 평균 (Spectral Envelope)
        avg_magnitude = magnitude.mean(axis=1)

        # 피크 개수 (High harmonicity check)
        # Lower threshold for better sensitivity to harmonics
        peaks, _ = find_peaks(avg_magnitude, height=np.max(avg_magnitude) * 0.05)

        # 스펙트럼 중심 (Spectral Centroid)
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0].mean()

        # 분류 규칙 (Simple heuristic based on design doc)
        if len(peaks) > 8 and centroid < 2000:
            return "voice"
        elif len(peaks) > 4:
            return "instrument"
        else:
            return "noise"
