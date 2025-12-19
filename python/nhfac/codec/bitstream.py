import struct
import zlib
import numpy as np


class NHFACBitstream:
    """
    Custom Bitstream for NHFAC (Hartley-based).
    Header Format (44 bytes):
    - Magic: 'NHFC' (4 bytes)
    - Version: 1 (2 bytes)
    - Sample Rate: (4 bytes)
    - Original Length: (8 bytes)
    - Pad Length: (4 bytes)
    - Frame Size: (4 bytes)
    - Degree: (1 byte)
    - Harmonics: (1 byte)
    - Factor: (4 bytes - float)
    - Q-Scale: (4 bytes - float)
    - H-Stream Length: (4 bytes)
    - T-Stream Length: (4 bytes)
    """

    HEADER_FORMAT = "<4sH I Q I I B B f f I I"
    MAGIC = b"NHFC"
    VERSION = 1

    @staticmethod
    def pack(encoded_data):
        h_stream = encoded_data["h_stream"]
        t_stream = encoded_data["t_stream"]

        header = struct.pack(
            NHFACBitstream.HEADER_FORMAT,
            NHFACBitstream.MAGIC,
            NHFACBitstream.VERSION,
            encoded_data["sr"],
            encoded_data["orig_len"],
            encoded_data["pad_len"],
            encoded_data.get("frame_size", 1024),
            encoded_data["degree"],
            encoded_data["n_harmonics"],
            encoded_data["factor"],
            encoded_data["q_scale"],
            len(h_stream),
            len(t_stream),
        )

        return header + h_stream + t_stream

    @staticmethod
    def unpack(binary_data):
        header_size = struct.calcsize(NHFACBitstream.HEADER_FORMAT)
        header = struct.unpack(NHFACBitstream.HEADER_FORMAT, binary_data[:header_size])

        if header[0] != NHFACBitstream.MAGIC:
            raise ValueError("Invalid NHFAC magic number")

        h_len = header[10]
        t_len = header[11]

        h_start = header_size
        h_end = h_start + h_len
        t_start = h_end
        t_end = t_start + t_len

        h_stream = binary_data[h_start:h_end]
        t_stream = binary_data[t_start:t_end]

        # Calculate actual n_frames for h_shape
        frame_size = header[5]
        h_decomp = zlib.decompress(h_stream)
        n_frames = len(h_decomp) // (frame_size * 2)

        return {
            "sr": header[2],
            "orig_len": header[3],
            "pad_len": header[4],
            "frame_size": frame_size,
            "degree": header[6],
            "n_harmonics": header[7],
            "factor": header[8],
            "q_scale": header[9],
            "h_stream": h_stream,
            "t_stream": t_stream,
            "signal_type": "unknown",
            "h_shape": (n_frames, frame_size),
        }
