import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class AdaptiveRegression:
    def __init__(self, signal, degree=4, n_harmonics=2, use_gpu=True):
        self.signal = signal
        self.degree = degree
        self.n_harmonics = n_harmonics
        self.xp = cp if (use_gpu and GPU_AVAILABLE) else np

    def _get_basis_matrix(self, window_size):
        """Create a hybrid basis: Polynomial + Sinusoidal (Harmonics)"""
        t = self.xp.linspace(-1, 1, window_size)
        basis = []

        # Polynomial components
        for d in range(self.degree + 1):
            basis.append(t**d)

        # Sinusoidal components (for periodic trends)
        for h in range(1, self.n_harmonics + 1):
            basis.append(self.xp.sin(h * np.pi * t))
            basis.append(self.xp.cos(h * np.pi * t))

        return self.xp.vstack(basis).T

    def fit_sliding_window(self, window_size=1024, step=512, decay=0.8):
        """
        Optimized Adaptive Regression.
        """
        # Move signal to GPU if needed
        sig = self.xp.asarray(self.signal)
        n = len(sig)
        model_signal = self.xp.zeros(n)
        overlap_weight = self.xp.zeros(n)

        A = self._get_basis_matrix(window_size)
        n_coeffs = A.shape[1]

        # Precompute Solver State
        lam = 0.05
        i_reg = lam * self.xp.eye(n_coeffs)
        # Precompute (A.T @ A + lambda*I)^-1 @ A.T
        # This turns the solve into a single matrix-vector multiply
        solver_matrix = self.xp.linalg.solve(A.T @ A + i_reg, A.T)

        thetas = []
        window = self.xp.hanning(window_size)

        current_theta = None

        # Process frames
        for i in range(0, n - window_size + 1, step):
            w = sig[i : i + window_size]
            theta_new = solver_matrix @ w

            if current_theta is None:
                current_theta = theta_new
            else:
                current_theta = decay * current_theta + (1 - decay) * theta_new

            thetas.append(current_theta.copy())

            # Reconstruct and overlap-add
            w_rec = (A @ current_theta) * window
            model_signal[i : i + window_size] += w_rec
            overlap_weight[i : i + window_size] += window

        overlap_weight[overlap_weight < 1e-6] = 1.0
        model_signal /= overlap_weight

        # Transfer back to CPU for storage
        res = sig - model_signal
        if self.xp != np:
            return [cp.asnumpy(t) for t in thetas], cp.asnumpy(res)
        return thetas, res

    def reconstruct_from_thetas(self, thetas, n_signal, window_size=1024, step=512):
        """Vectorized reconstruction from thetas"""
        thetas_gpu = self.xp.asarray(thetas)
        model_signal = self.xp.zeros(n_signal)
        overlap_weight = self.xp.zeros(n_signal)

        A = self._get_basis_matrix(window_size)
        window = self.xp.hanning(window_size)

        for idx, i in enumerate(range(0, n_signal - window_size + 1, step)):
            if idx >= len(thetas_gpu):
                break
            w_rec = (A @ thetas_gpu[idx]) * window
            model_signal[i : i + window_size] += w_rec
            overlap_weight[i : i + window_size] += window

        overlap_weight[overlap_weight < 1e-6] = 1.0
        model_signal /= overlap_weight

        if self.xp != np:
            return cp.asnumpy(model_signal)
        return model_signal
