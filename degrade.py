"""
Codec degradation module — simulates phone/VoIP codec round-trip distortion.
Note: real network jitter/packet loss is not simulated here.
"""

import os
import shutil
import subprocess
import sys
import tempfile

# (encode_args, intermediate_ext, decode_args)
_CODECS = {
    "g711":     (["-acodec", "pcm_mulaw",    "-ar", "8000"],  ".wav",  ["-acodec", "pcm_s16le", "-ar", "16000"]),
    "g726":     (["-acodec", "adpcm_g726",   "-ar", "8000", "-b:a", "32k"],  ".wav",  ["-acodec", "pcm_s16le", "-ar", "16000"]),
    "mp3_low":  (["-acodec", "libmp3lame",   "-b:a", "32k"],  ".mp3",  ["-acodec", "pcm_s16le", "-ar", "16000"]),
    "opus_low": (["-acodec", "libopus",      "-b:a", "16k"],  ".opus", ["-acodec", "pcm_s16le", "-ar", "16000"]),
}


def apply_codec(input_path: str, codec: str = "g711", output_path: str = None) -> str:
    """Encode then decode audio through a phone codec, returning a 16 kHz WAV.

    codec options: "g711", "g726", "mp3_low", "opus_low".
    If output_path is None, writes to a temp file inside a new mkdtemp directory.
    Returns the path to the degraded file.
    """
    if codec not in _CODECS:
        raise ValueError(f"Unknown codec '{codec}'. Choose from: {list(_CODECS)}")

    encode_args, ext, decode_args = _CODECS[codec]
    tmp_dir = tempfile.mkdtemp(prefix="degrade_")
    intermediate = os.path.join(tmp_dir, f"intermediate{ext}")
    if output_path is None:
        output_path = os.path.join(tmp_dir, f"{codec}.wav")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path] + encode_args + [intermediate],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", intermediate] + decode_args + [output_path],
            check=True, capture_output=True, timeout=120,
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)   # a failed codec must not leak its temp dir
        raise
    os.remove(intermediate)
    if not output_path.startswith(tmp_dir):
        os.rmdir(tmp_dir)          # caller chose the destination — don't leak the temp dir
    return output_path


def degrade_all(input_path: str) -> dict:
    """Run all four codecs on input_path.

    Returns {"clean": input_path, "g711": path, "g726": path,
             "mp3_low": path, "opus_low": path}.
    """
    results = {"clean": input_path}
    try:
        for codec in _CODECS:
            results[codec] = apply_codec(input_path, codec)
    except Exception:
        cleanup(results)                             # drop the codecs that did succeed
        raise
    return results


def cleanup(tmp_dir) -> None:
    """Delete temp dirs produced by apply_codec / degrade_all.
    Accepts a single path str or the dict returned by degrade_all."""
    if isinstance(tmp_dir, dict):
        for k, p in tmp_dir.items():
            if k != "clean":
                shutil.rmtree(os.path.dirname(p), ignore_errors=True)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg is not installed or not on PATH. "
              "Install ffmpeg before using this module.")
        sys.exit(0)
    print("ffmpeg found:", shutil.which("ffmpeg"))
    print("Available codecs:", list(_CODECS))
