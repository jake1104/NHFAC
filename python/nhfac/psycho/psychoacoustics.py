import numpy as np


def hz_to_bark(f):
    """Converts frequency in Hz to Bark scale."""
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def get_ath(freqs):
    """Absolute Threshold of Hearing (ATH) in dB SPL."""
    f = np.array(freqs) / 1000.0  # kHz
    f = np.where(f < 0.1, 0.1, f)
    ath = 3.64 * (f**-0.8) - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 10**-3 * (f**4)
    return ath


def apply_masking(h_coeffs, sr):
    """
    Apply advanced psychoacoustic masking to Hartley coefficients using Bark scale spreading.
    """
    N = len(h_coeffs)
    half = N // 2
    freqs = np.fft.rfftfreq(N, 1 / sr)
    barks = hz_to_bark(freqs)

    # 1. Compute Magnitude Spectrum (dB SPL)
    mag = np.zeros(half + 1)
    mag[0] = np.abs(h_coeffs[0])
    mag[1:half] = np.sqrt(
        (h_coeffs[1:half] ** 2 + h_coeffs[N - 1 : half : -1] ** 2) / 2.0
    )
    mag[half] = np.abs(h_coeffs[half])

    eps = 1e-10
    spl = 20 * np.log10(mag + eps) + 96  # Assume 96dB dynamic range

    # 2. Simultaneous Masking (Spreading Function)
    # Simple spreading: -25dB/Bark slope for lower, -10dB/Bark for higher
    masking_threshold = np.full_like(spl, -np.inf)

    # For each bin, calculate its masking effect on other bins
    for i in range(len(spl)):
        if spl[i] < -10:
            continue

        # Masking effect of bin i
        mask_level = spl[i] - 15  # Tonal masker usually 15-20dB below peak

        # Bark distance
        dist = barks - barks[i]

        # Spreading function (Triangular approximation in Bark domain)
        spreading = np.where(dist >= 0, -25 * dist, 10 * dist)
        bin_mask = mask_level + spreading

        masking_threshold = np.maximum(masking_threshold, bin_mask)

    # 3. Combine with ATH
    ath = get_ath(freqs)
    final_threshold_db = np.maximum(masking_threshold, ath)

    # Allow some noise floor (Optional: SMR adjustment)
    final_threshold_db -= 3.0  # Slightly more aggressive

    threshold_lin = 10 ** ((final_threshold_db - 96) / 20)

    # 4. Apply Masking
    masked_h = np.zeros_like(h_coeffs)

    # Keep components above threshold
    keep_indices = mag > threshold_lin

    if keep_indices[0]:
        masked_h[0] = h_coeffs[0]
    if keep_indices[half]:
        masked_h[half] = h_coeffs[half]

    masked_h[1:half] = np.where(keep_indices[1:half], h_coeffs[1:half], 0)
    masked_h[N - 1 : half : -1] = np.where(
        keep_indices[1:half], h_coeffs[N - 1 : half : -1], 0
    )

    return masked_h.astype(np.float32)
