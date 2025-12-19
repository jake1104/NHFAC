import numpy as np


class NHFACMetrics:
    @staticmethod
    def calculate_all(original, reconstructed, residual_orig, sr):
        """
        Calculates a suite of metrics tailored for NHFAC.
        Includes Global SNR, Segmental SNR, LSD, and Residual Fidelity.
        """
        n = min(len(original), len(reconstructed))
        s = original[:n]
        hat_s = reconstructed[:n]
        noise = s - hat_s

        # 1. Global SNR (Overall system performance)
        snr_global = 10 * np.log10(np.mean(s**2) / (np.mean(noise**2) + 1e-10))

        # 2. Segmental SNR (Captures performance in non-stationary regions)
        frame_len = int(0.02 * sr)  # 20ms frames
        ssnr_values = []
        for i in range(0, n - frame_len, frame_len // 2):
            f_s = s[i : i + frame_len]
            f_n = noise[i : i + frame_len]
            p_s = np.mean(f_s**2)
            p_n = np.mean(f_n**2)
            if p_s > 1e-7:  # Only active regions
                ssnr_values.append(10 * np.log10(p_s / (p_n + 1e-10)))
        ssnr = np.mean(ssnr_values) if ssnr_values else snr_global

        # 3. Log-Spectral Distance (LSD) - Spectral envelope fidelity
        # Closer to human perception of timbral accuracy
        lsd = NHFACMetrics._calculate_lsd(s, hat_s, sr)

        # 4. Residual SNR (Quantization fidelity of the HF stage)
        # This is specific to NHFAC: how much noise was added to the residual
        res_n = min(len(residual_orig), len(noise))
        snr_residual = 10 * np.log10(
            np.mean(residual_orig[:res_n] ** 2) / (np.mean(noise[:res_n] ** 2) + 1e-10)
        )

        return {
            "snr_global": snr_global,
            "ssnr": ssnr,
            "lsd": lsd,
            "snr_residual": snr_residual,
        }

    @staticmethod
    def _calculate_lsd(s, hat_s, sr):
        n_fft = 2048
        hop = n_fft // 2
        lsd_segments = []

        for i in range(0, len(s) - n_fft, hop):
            f_s = np.fft.rfft(s[i : i + n_fft] * np.hanning(n_fft))
            f_hat = np.fft.rfft(hat_s[i : i + n_fft] * np.hanning(n_fft))

            mag_s = 20 * np.log10(np.abs(f_s) + 1e-10)
            mag_hat = 20 * np.log10(np.abs(f_hat) + 1e-10)

            lsd_segments.append(np.sqrt(np.mean((mag_s - mag_hat) ** 2)))

        return np.mean(lsd_segments) if lsd_segments else 0.0
