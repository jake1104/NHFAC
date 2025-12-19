import soundfile as sf
import librosa
import numpy as np


class AudioIO:
    @staticmethod
    def read(path, sr=48000):
        """오디오 읽기 및 리샘플링 (Audio Loading and Resampling)"""
        data, samplerate = sf.read(path, dtype="float32")

        # Stereo to Mono if needed (simplified for codec)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        if sr != samplerate:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=sr)

        return data, sr

    @staticmethod
    def write(path, signal, sr):
        """오디오 저장 (Audio Saving)"""
        sf.write(path, signal, sr, subtype="FLOAT")
