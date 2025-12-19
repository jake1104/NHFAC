import numpy as np
import sounddevice as sd
import queue
import threading
from ..codec.encoder import NHFACEncoder
from ..codec.decoder import NHFACDecoder


class NHFACRealTimeStreamer:
    """
    Real-time Audio Streaming using NHFAC Codec.
    Captures input, encodes, decodes (loopback), and plays back.
    """

    def __init__(self, sr=48000, frame_size=2048):
        self.sr = sr
        self.frame_size = frame_size
        self.encoder = NHFACEncoder(sr=sr, frame_size=frame_size)
        self.decoder = NHFACDecoder(sr=sr, frame_size=frame_size)

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_running = False
        self.thread = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Status: {status}")

        # 1. Put incoming audio to queue
        self.input_queue.put(indata.copy().flatten())

        # 2. Get decoded audio from queue for playback
        try:
            # We need to provide exactly 'frames' samples
            data = self.output_queue.get_nowait()
            if len(data) < frames:
                outdata[: len(data), 0] = data
                outdata[len(data) :, 0] = 0
            else:
                outdata[:, 0] = data[:frames]
        except queue.Empty:
            outdata.fill(0)

    def _processing_loop(self):
        """
        Background thread for CPU-intensive encoding/decoding.
        """
        buffer = np.array([], dtype=np.float32)

        while self.is_running:
            try:
                # Accumulate enough samples for a frame
                chunk = self.input_queue.get(timeout=1.0)
                buffer = np.concatenate((buffer, chunk))

                if len(buffer) >= self.frame_size:
                    frame = buffer[: self.frame_size]
                    buffer = buffer[self.frame_size // 2 :]  # Overlap

                    # --- NHFAC PIPELINE ---
                    # 1. Encode
                    encoded = self.encoder.encode(frame)
                    # 2. Decode
                    reconstructed = self.decoder.decode(encoded)
                    # ----------------------

                    # Push to output queue (take the second half for OLA simulation or just simple chunking)
                    # For real OLA in streaming, we'd need more careful state management.
                    # This is a simplified low-latency approximation.
                    self.output_queue.put(reconstructed[-self.frame_size // 2 :])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.frame_size // 2,
            channels=1,
            callback=self._audio_callback,
        )
        self.stream.start()
        print("NHFAC Real-time stream started...")

    def stop(self):
        self.is_running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if self.thread:
            self.thread.join()
        print("NHFAC Real-time stream stopped.")


if __name__ == "__main__":
    # Test script
    import time

    streamer = NHFACRealTimeStreamer(sr=44100, frame_size=2048)
    streamer.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        streamer.stop()
