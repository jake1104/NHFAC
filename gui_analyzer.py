import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nhfac.codec.encoder import NHFACEncoder
from nhfac.codec.decoder import NHFACDecoder
from nhfac.io.soundfile_io import AudioIO
from nhfac.core.metrics import NHFACMetrics
from tkinter import filedialog
import os
import time
import sounddevice as sd
import zlib
from nhfac.codec.bitstream import NHFACBitstream
from nhfac.api.ai_features import NHFACFeatureExtractor

UPPER_RIGHT = "upper right"
AWAITING_PROC = "Awaiting Processing..."
PROCESS_TEXT = "Process NHFAC"
N_HARM_DEFAULT = 4

# Globals for better graph visibility
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["legend.facecolor"] = "#2d2d2d"
plt.rcParams["legend.edgecolor"] = "white"


class NHFACVisualizer(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NHFAC Visual Analyzer (Nonlinear Hartley-Fourier Audio Codec)")
        self.geometry("1400x950")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.sr = 48000
        self.signal = None
        self.reconstructed = None
        self.encoded_data = None
        self.file_path = None
        self.is_processing = False

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar (Commands + Log)
        self.sidebar_left = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_left.grid(row=0, column=0, sticky="nsew")

        # Right Sidebar (Metrics + Playback)
        self.sidebar_right = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_right.grid(row=0, column=2, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar_left,
            text="NHFAC Core",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.logo_label.pack(pady=20)

        self.load_btn = ctk.CTkButton(
            self.sidebar_left, text="Load File", command=self.load_file, height=40
        )
        self.load_btn.pack(padx=20, pady=5, fill="x")

        self.process_btn = ctk.CTkButton(
            self.sidebar_left,
            text="Process NHFAC",
            command=self.process_codec,
            state="disabled",
            height=40,
            fg_color="#2ecc71",
            hover_color="#27ae60",
        )
        self.process_btn.pack(padx=20, pady=5, fill="x")

        self.save_btn = ctk.CTkButton(
            self.sidebar_left,
            text="Save .nhfac",
            command=self.save_nhfac,
            state="disabled",
            height=40,
            fg_color="#34495e",
        )
        self.save_btn.pack(padx=20, pady=5, fill="x")

        self.save_wav_btn = ctk.CTkButton(
            self.sidebar_left,
            text="Export .wav",
            command=self.save_wav,
            state="disabled",
            height=40,
            fg_color="#34495e",
        )
        self.save_wav_btn.pack(padx=20, pady=5, fill="x")

        self.export_ai_btn = ctk.CTkButton(
            self.sidebar_left,
            text="Export AI Features",
            command=self.export_ai_features,
            state="disabled",
            fg_color="#a832a4",
            hover_color="#82207d",
            height=40,
        )
        self.export_ai_btn.pack(padx=20, pady=5, fill="x")

        self.reset_btn = ctk.CTkButton(
            self.sidebar_left,
            text="Reset",
            command=self.reset_all,
            height=32,
            fg_color="#95a5a6",
        )
        self.reset_btn.pack(padx=20, pady=5, fill="x")

        # Metrics Section in Right Sidebar
        self.metrics_label = ctk.CTkLabel(
            self.sidebar_right,
            text="Quality Metrics",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.metrics_label.pack(pady=(30, 10))

        self.status_frame = ctk.CTkFrame(
            self.sidebar_right, fg_color="#1e272e", corner_radius=10
        )
        self.status_frame.pack(pady=10, fill="x", padx=20)

        # Playback Section in Right Sidebar
        self.playback_label = ctk.CTkLabel(
            self.sidebar_right,
            text="Playback",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.playback_label.pack(pady=(20, 10))

        self.playback_frame = ctk.CTkFrame(
            self.sidebar_right, fg_color="#2c3e50", corner_radius=10
        )
        self.playback_frame.pack(pady=10, padx=20, fill="x")

        self.play_label = ctk.CTkLabel(
            self.playback_frame,
            text="Audio Playback",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.play_label.pack(pady=(5, 0))

        self.play_org_btn = ctk.CTkButton(
            self.playback_frame,
            text="Play Source Audio",
            command=lambda: self.play_audio(self.signal, "Source"),
            state="disabled",
            height=32,
            fg_color="#7f8c8d",
        )
        self.play_org_btn.pack(padx=10, pady=5, fill="x")

        self.play_rec_btn = ctk.CTkButton(
            self.playback_frame,
            text="Play NHFAC Output",
            command=lambda: self.play_audio(self.reconstructed, "NHFAC Output"),
            state="disabled",
            height=32,
            fg_color="#f39c12",
        )
        self.play_rec_btn.pack(padx=10, pady=5, fill="x")

        self.stop_btn = ctk.CTkButton(
            self.playback_frame,
            text="⏹ Stop",
            command=self.stop_audio,
            height=32,
            fg_color="#c0392b",
        )
        self.stop_btn.pack(padx=10, pady=(5, 10), fill="x")

        self.snr_label = ctk.CTkLabel(
            self.status_frame,
            text="SNR: -- dB",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.snr_label.pack(pady=2)

        self.ssnr_label = ctk.CTkLabel(
            self.status_frame, text="SSNR: -- dB", font=ctk.CTkFont(size=14)
        )
        self.ssnr_label.pack(pady=2)

        self.res_snr_label = ctk.CTkLabel(
            self.status_frame, text="Res-SNR: -- dB", font=ctk.CTkFont(size=12)
        )
        self.res_snr_label.pack(pady=2)

        self.lsd_label = ctk.CTkLabel(
            self.status_frame, text="LSD: --", font=ctk.CTkFont(size=12)
        )
        self.lsd_label.pack(pady=2)

        self.type_label = ctk.CTkLabel(
            self.status_frame,
            text="Type: --",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        self.type_label.pack(pady=10)

        # Analysis Log Area in Left Sidebar
        self.log_label = ctk.CTkLabel(
            self.sidebar_left,
            text="Process Log",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.log_label.pack(pady=(20, 0))

        self.log_textbox = ctk.CTkTextbox(
            self.sidebar_left, font=ctk.CTkFont(size=11), fg_color="#1a1a1a"
        )
        self.log_textbox.pack(padx=10, pady=10, fill="both", expand=True)
        self.log_textbox.configure(state="disabled")

        # Main Plot Area
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Waveform Analysis")
        self.tabview.add("Spectral Analysis")
        self.tabview.add("AI Latent Features")

        # Waveform Tab Plots
        self.fig_wave, self.axs_wave = plt.subplots(
            3, 1, facecolor="#1a1a1a", layout="constrained"
        )
        for ax in self.axs_wave:
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.title.set_fontsize(10)

        wave_tab = self.tabview.tab("Waveform Analysis")
        wave_tab.grid_columnconfigure(0, weight=1)
        wave_tab.grid_rowconfigure(0, weight=1)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=wave_tab)
        self.canvas_wave.get_tk_widget().grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5
        )

        # Spectral Tab Plots
        self.fig_spec, self.axs_spec = plt.subplots(
            2, 1, facecolor="#1a1a1a", layout="constrained"
        )
        for ax in self.axs_spec:
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.title.set_fontsize(10)

        spec_tab = self.tabview.tab("Spectral Analysis")
        spec_tab.grid_columnconfigure(0, weight=1)
        spec_tab.grid_rowconfigure(0, weight=1)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_tab)
        self.canvas_spec.get_tk_widget().grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5
        )

        # AI Tab Plots
        self.fig_ai, self.axs_ai = plt.subplots(
            2, 1, facecolor="#1a1a1a", layout="constrained"
        )
        for ax in self.axs_ai:
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="white", labelsize=9)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.title.set_fontsize(10)

        ai_tab = self.tabview.tab("AI Latent Features")
        ai_tab.grid_columnconfigure(0, weight=1)
        ai_tab.grid_rowconfigure(0, weight=1)
        self.canvas_ai = FigureCanvasTkAgg(self.fig_ai, master=ai_tab)
        self.canvas_ai.get_tk_widget().grid(
            row=0, column=0, sticky="nsew", padx=5, pady=5
        )

    def reset_all(self):
        self.signal = None
        self.reconstructed = None
        self.encoded_data = None
        self.process_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.save_wav_btn.configure(state="disabled")
        self.export_ai_btn.configure(state="disabled")
        self.play_org_btn.configure(state="disabled")
        self.play_rec_btn.configure(state="disabled")
        self.is_processing = False
        self.stop_audio()
        self.clear_log()
        for ax in self.axs_wave:
            ax.clear()
        for ax in self.axs_spec:
            ax.clear()
        self.canvas_wave.draw()
        self.canvas_spec.draw()

    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio/Codec Files", "*.wav *.flac *.nhfac")]
        )
        if path:
            self.reset_all()
            self.file_path = path
            if path.endswith(".nhfac"):
                with open(path, "rb") as f:
                    binary_data = f.read()
                self.encoded_data = NHFACBitstream.unpack(binary_data)
                self.sr = self.encoded_data.get("sr", 48000)
                self.signal = None  # No original signal available
                self.log(f"Opened: {os.path.basename(path)}")
                self.log("Decrypting bitstream for reconstruction...")

                # Auto-start decoding process for immediate feedback
                self.process_codec()
            else:
                self.signal, self.sr = AudioIO.read(path, sr=self.sr)
                self.log(
                    f"Loaded audio: {os.path.basename(path)} ({len(self.signal)/self.sr:.2f}s)"
                )
                self.update_plots(step=0)
                self.process_btn.configure(state="normal")
                self.play_org_btn.configure(state="normal")

    def load_test_signal(self):
        t = np.linspace(0, 1.0, int(self.sr * 1.0), endpoint=False)
        trend = 0.2 * (t**2) + 0.1 * t
        sine = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        rng = np.random.default_rng(42)
        noise = 0.03 * rng.normal(size=len(t))
        self.signal = trend + sine + noise
        self.clear_log()
        self.log("Synthetic signal generated.")
        self.update_plots(step=0)
        self.process_btn.configure(state="normal")
        self.play_org_btn.configure(state="normal")

    def play_audio(self, signal, label):
        if signal is not None and len(signal) > 0:
            self.log(f"Playing {label}...")
            sd.stop()
            sd.play(signal, self.sr)
        else:
            self.log(f"No {label} audio available yet.")

    def stop_audio(self):
        sd.stop()
        self.log("Playback stopped.")

    def log(self, text):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")
        self.update()

    def clear_log(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def save_nhfac(self):
        if self.encoded_data:
            path = filedialog.asksaveasfilename(
                defaultextension=".nhfac",
                filetypes=[("NHFAC Compressed Audio", "*.nhfac")],
            )
            if path:
                with open(path, "wb") as f:
                    binary_data = NHFACBitstream.pack(self.encoded_data)
                    f.write(binary_data)

                size_kb = os.path.getsize(path) / 1024
                self.log(f"Saved: {os.path.basename(path)} ({size_kb:.1f} KB)")

    def save_wav(self):
        if self.reconstructed is not None:
            path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("Waveform Audio", "*.wav")],
            )
            if path:
                AudioIO.write(path, self.reconstructed, self.sr)
                self.log(f"Successfully exported to {os.path.basename(path)}")

    def process_codec(self):
        if self.is_processing:
            return

        self.is_processing = True
        self.clear_log()
        self.log("Starting NHFAC pipeline...")
        self.process_btn.configure(state="disabled", text="Processing...")

        try:
            # Ensure we have something to process
            if self.signal is None and self.encoded_data is None:
                self.log("Error: No data loaded.")
                self.process_btn.configure(state="disabled", text=PROCESS_TEXT)
                return

            decoder = NHFACDecoder(sr=self.sr)

            # Step 1 & 2: Encoding / Data Check
            if self.signal is not None:
                self.log("Step 1: Signal Classification & Profiling...")
                time.sleep(0.3)
                self.log("Step 2: Performing Adaptive Regression...")
                encoder = NHFACEncoder(sr=self.sr)
                self.encoded_data = encoder.encode(self.signal)
                self.update_plots(step=1)
                time.sleep(0.5)
            else:
                self.log("Step 1: Bitstream profiling...")
                self.log("Step 2: NHFAC data already loaded.")
                time.sleep(0.3)

            # CRITICAL CHECK for self.encoded_data
            if self.encoded_data is None or (
                "h_stream" not in self.encoded_data
                and "h_matrix" not in self.encoded_data
                and "frames" not in self.encoded_data
            ):
                self.log("Error: Encoded data is missing or invalid.")
                return

            # Step 3: Hartley Transform Analysis
            self.log("Step 3: Calculating Hartley Transform...")
            if "h_stream" in self.encoded_data:
                n_frames = self.encoded_data["h_shape"][0]
            elif "h_matrix" in self.encoded_data:
                n_frames = len(self.encoded_data["h_matrix"])
            else:
                n_frames = len(self.encoded_data.get("frames", []))
            self.log(f" -> Processed {n_frames} frames via optimized OLA.")
            self.update_plots(step=2)
            time.sleep(0.5)

            # Step 4: Decoding
            self.log("Step 4: Reconstructing signal via Inverse Hartley...")
            self.reconstructed = decoder.decode(self.encoded_data)
            time.sleep(0.3)

            # Final Analysis
            self.log("Step 5: Finalizing metrics.")

            if self.signal is not None:
                metrics = NHFACMetrics.calculate_all(
                    self.signal,
                    self.reconstructed,
                    self.encoded_data["residual"],
                    self.sr,
                )

                self.snr_label.configure(
                    text=f"Global SNR: {metrics['snr_global']:.2f} dB"
                )
                self.ssnr_label.configure(text=f"Seg-SNR: {metrics['ssnr']:.2f} dB")
                self.res_snr_label.configure(
                    text=f"Res-SNR: {metrics['snr_residual']:.2f} dB"
                )
                self.lsd_label.configure(text=f"LSD Error: {metrics['lsd']:.3f}")
            else:
                self.snr_label.configure(text="SNR: N/A (Source is .nhfac)")
                self.ssnr_label.configure(text="SSNR: N/A")
                self.res_snr_label.configure(text="Res-SNR: N/A")
                self.lsd_label.configure(text="LSD: N/A")

            self.type_label.configure(
                text=f"Detected: {self.encoded_data['signal_type']}"
            )
            self.process_btn.configure(state="normal", text=PROCESS_TEXT)
            self.save_btn.configure(state="normal")
            self.save_wav_btn.configure(state="normal")
            self.export_ai_btn.configure(state="normal")
            self.play_rec_btn.configure(state="normal")
            self.log("Conversion complete.")
            self.update_plots(step=3)
            self.canvas_wave.draw()
            self.canvas_spec.draw()
            self.canvas_ai.draw()

        except Exception as e:
            self.log(f"Process failed: {str(e)}")
            self.process_btn.configure(state="normal")
        finally:
            self.is_processing = False
            self.process_btn.configure(text=PROCESS_TEXT)
            if self.signal is not None:
                self.process_btn.configure(state="normal")
            elif self.encoded_data is not None:
                self.process_btn.configure(state="normal")

    def update_plots(self, step=0):
        for ax in self.axs_wave:
            ax.clear()
            ax.set_facecolor("#2d2d2d")

        target_viz_points = 5000
        trend = None

        # Determine total length for x-axis scaling
        if self.signal is not None:
            s_len = len(self.signal)
        elif self.encoded_data:
            s_len = self.encoded_data.get("orig_len", 1)
        else:
            s_len = 1

        def downsample(data, target_points):
            if len(data) <= target_points:
                return data, np.linspace(0, len(data) / self.sr, len(data))
            step = len(data) // target_points
            return data[::step], np.linspace(0, len(data) / self.sr, len(data[::step]))

        if step >= 1 and self.encoded_data:
            from nhfac.core.regression import AdaptiveRegression

            pad_len = self.encoded_data.get("pad_len", 0)
            n_harm = self.encoded_data.get("n_harmonics", N_HARM_DEFAULT)

            degree = self.encoded_data.get("degree")
            if degree is None:
                thetas = self.encoded_data.get("thetas", [[]])
                degree = len(thetas[0]) - 1 - (2 * n_harm)

            if "h_stream" in self.encoded_data:
                n_frames = self.encoded_data["h_shape"][0]
                t_bytes = zlib.decompress(self.encoded_data["t_stream"])
                n_harm = self.encoded_data.get("n_harmonics", N_HARM_DEFAULT)
                degree = self.encoded_data.get("degree", 0)
                n_theta_coeffs = (degree + 1) + (2 * n_harm)
                thetas = np.frombuffer(t_bytes, dtype=np.float32).reshape(
                    (-1, n_theta_coeffs)
                )
            elif "h_matrix" in self.encoded_data:
                n_frames = len(self.encoded_data["h_matrix"])
                thetas = self.encoded_data["thetas"]
            else:
                n_frames = len(self.encoded_data.get("frames", []))
                thetas = self.encoded_data.get("thetas")

            reg = AdaptiveRegression(None, degree=degree, n_harmonics=n_harm)
            n_padded = (n_frames - 1) * (
                self.encoded_data.get("frame_size", 1024) // 2
            ) + self.encoded_data.get("frame_size", 1024)
            full_trend = reg.reconstruct_from_thetas(thetas, n_padded)
            trend = full_trend[pad_len : pad_len + s_len]

        # 1. Regression Analysis
        self.axs_wave[0].set_title(
            "Step 1: Nonlinear Regression Analysis (Whole Signal)"
        )
        if self.signal is not None:
            sig_ds, t_ds = downsample(self.signal, target_viz_points)
            self.axs_wave[0].plot(
                t_ds, sig_ds, label="Input", color="#3498db", alpha=0.5
            )
        else:
            self.axs_wave[0].text(0.5, 0.5, "Source: .nhfac", color="gray", ha="center")

        if step >= 1 and trend is not None:
            trend_ds, t_ds = downsample(trend, target_viz_points)
            self.axs_wave[0].plot(
                t_ds, trend_ds, label="Trend", color="#e74c3c", linewidth=1.5
            )
        self.axs_wave[0].legend(loc=UPPER_RIGHT)
        self.axs_wave[0].set_xlabel("Time (s)")

        # 2. Residual View
        self.axs_wave[1].set_title("Step 2: Residual Signal (Compressed View)")
        if step >= 2:
            if self.signal is not None and trend is not None:
                res = self.signal - trend
                res_ds, t_ds = downsample(res, target_viz_points)
                self.axs_wave[1].plot(t_ds, res_ds, color="#2ecc71")
            elif self.encoded_data and "residual" in self.encoded_data:
                res_stored = self.encoded_data["residual"][pad_len : pad_len + s_len]
                res_ds, t_ds = downsample(res_stored, target_viz_points)
                self.axs_wave[1].plot(t_ds, res_ds, color="#2ecc71", alpha=0.7)
        else:
            self.axs_wave[1].text(0.5, 0.5, AWAITING_PROC, color="gray", ha="center")
        self.axs_wave[1].set_xlabel("Time (s)")

        # 3. Final Output Comparison
        self.axs_wave[2].set_title("Step 3: Reconstruction (Full Length)")
        if step >= 3:
            if self.signal is not None:
                sig_ds, t_ds = downsample(self.signal, target_viz_points)
                self.axs_wave[2].plot(
                    t_ds, sig_ds, label="Original", color="gray", alpha=0.4
                )
            rec_ds, t_ds = downsample(self.reconstructed, target_viz_points)
            self.axs_wave[2].plot(t_ds, rec_ds, label="NHFAC Output", color="#f1c40f")
            self.axs_wave[2].legend(loc=UPPER_RIGHT)
        else:
            self.axs_wave[2].text(0.5, 0.5, AWAITING_PROC, color="gray", ha="center")
        self.axs_wave[2].set_xlabel("Time (s)")

        self.canvas_wave.draw()

        # Update Spectral Plots
        for ax in self.axs_spec:
            ax.clear()
            ax.set_facecolor("#2d2d2d")

        if self.signal is not None or self.reconstructed is not None:
            # 1. Power Spectral Density (Global Average)
            self.axs_spec[0].set_title("Global Power Spectral Density (Whole Signal)")
            n_fft = 4096

            def get_avg_psd(sig, sr, n_fft):
                if sig is None or len(sig) == 0:
                    return None, None
                # Use scipy.signal.welch for a clean average PSD across the whole signal
                from scipy.signal import welch

                f, pxx = welch(sig, sr, nperseg=n_fft)
                # Convert to dB
                return f, 10 * np.log10(pxx + 1e-12)

            # Plot Original
            freqs, psd_org = get_avg_psd(self.signal, self.sr, n_fft)
            if psd_org is not None:
                self.axs_spec[0].plot(
                    freqs, psd_org, label="Original", color="#3498db", alpha=0.7
                )

            # Plot Reconstructed
            if step >= 3 and self.reconstructed is not None:
                freqs_rec, psd_rec = get_avg_psd(self.reconstructed, self.sr, n_fft)
                if psd_rec is not None:
                    self.axs_spec[0].plot(
                        freqs_rec,
                        psd_rec,
                        label="NHFAC Output",
                        color="#f1c40f",
                        linestyle="--",
                        alpha=0.8,
                    )

            self.axs_spec[0].set_xlabel("Frequency (Hz)")
            self.axs_spec[0].set_ylabel("Average Magnitude (dB)")
            self.axs_spec[0].legend(loc=UPPER_RIGHT)
            self.axs_spec[0].grid(True, alpha=0.1)
            self.axs_spec[0].set_xlim(0, self.sr // 2)

            # 2. Hartley Coefficients (First Frame)
            self.axs_spec[1].set_title(
                "Hartley Coefficients (Residual Analysis - Frame 0)"
            )
            if step >= 3 and self.encoded_data:
                if "h_stream" in self.encoded_data:
                    h_bytes = zlib.decompress(self.encoded_data["h_stream"])
                    h_shape = self.encoded_data["h_shape"]
                    h_quant = np.frombuffer(h_bytes, dtype=np.int16).reshape(h_shape)
                    h_delta = (
                        h_quant[0].astype(np.float32) / self.encoded_data["q_scale"]
                    )
                    # Inverse log for visualization
                    factor = self.encoded_data["factor"]
                    h_coeffs = np.sign(h_delta) * (np.expm1(np.abs(h_delta)) / factor)
                elif "h_matrix" in self.encoded_data:
                    h_coeffs = self.encoded_data["h_matrix"][0]
                elif "frames" in self.encoded_data and self.encoded_data["frames"]:
                    h_coeffs = self.encoded_data["frames"][0]["h_coeffs"]
                else:
                    h_coeffs = None

                if h_coeffs is not None:
                    self.axs_spec[1].plot(h_coeffs, color="#2ecc71", linewidth=1)
                    self.axs_spec[1].set_xlabel("Coefficient Index")
                    self.axs_spec[1].set_ylabel("Amplitude")
            else:
                self.axs_spec[1].text(
                    0.5, 0.5, AWAITING_PROC, color="gray", ha="center"
                )

        self.canvas_spec.draw()

        # Update AI Plots
        for ax in self.axs_ai:
            ax.clear()
            ax.set_facecolor("#2d2d2d")

        if step >= 1 and self.encoded_data:
            self.axs_ai[0].set_title(
                "AI Feature 1: Structural Thetas (Latent Fingerprint)"
            )
            if "t_stream" in self.encoded_data:
                t_bytes = zlib.decompress(self.encoded_data["t_stream"])
                n_harm = self.encoded_data.get("n_harmonics", N_HARM_DEFAULT)
                degree = self.encoded_data.get("degree", 0)
                n_coeffs = (degree + 1) + (2 * n_harm)
                thetas = np.frombuffer(t_bytes, dtype=np.float32).reshape(
                    (-1, n_coeffs)
                )

                self.axs_ai[0].imshow(thetas.T, aspect="auto", cmap="magma")
                self.axs_ai[0].set_ylabel("Theta Index")
                self.axs_ai[0].set_xlabel("Frame Index")

            self.axs_ai[1].set_title(
                "AI Feature 2: High-Level Spectral Energy Trajectory"
            )
            if "h_stream" in self.encoded_data:
                h_bytes = zlib.decompress(self.encoded_data["h_stream"])
                h_quant = np.frombuffer(h_bytes, dtype=np.int16).reshape(
                    self.encoded_data["h_shape"]
                )
                energy = np.log10(
                    np.sum(h_quant.astype(np.float32) ** 2, axis=1) + 1e-6
                )
                self.axs_ai[1].plot(energy, color="#a832a4", linewidth=1.5)
                self.axs_ai[1].set_ylabel("Log Energy")
                self.axs_ai[1].set_xlabel("Frame Index")

        self.canvas_ai.draw()

    def export_ai_features(self):
        if self.signal is None:
            self.log("Load a file first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".npz", filetypes=[("Numpy Compressed", "*.npz")]
        )
        if save_path:
            try:
                self.log("Extracting AI features...")
                extractor = NHFACFeatureExtractor(sr=self.sr)
                output = extractor.export_to_numpy(self.signal, save_path)
                self.log(f"AI Features exported to: {output}")
            except Exception as e:
                self.log(f"Export failed: {str(e)}")


if __name__ == "__main__":
    app = NHFACVisualizer()
    app.mainloop()
