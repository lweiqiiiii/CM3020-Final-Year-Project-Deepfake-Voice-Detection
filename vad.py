"""Silero VAD front-end (pre-trained model #2) — keep only the speech in a call clip.

Call recordings carry ringing, hold silence and pauses; the detector should
score the voice, not the gaps. pip3 install silero-vad
"""
import threading

import torch
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

_model = None
_lock = threading.Lock()   # Silero is stateful (reset per call) — serialise concurrent Flask requests


def speech_only(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """(1, N) 16 kHz mono -> (1, M) speech-only. Returns input unchanged if no speech found."""
    global _model
    with _lock:
        if _model is None:
            _model = load_silero_vad()
        ts = get_speech_timestamps(wav[0], _model, sampling_rate=sr)
    if not ts:
        return wav
    return collect_chunks(ts, wav[0]).unsqueeze(0)


if __name__ == "__main__":
    sr = 16000
    t = torch.arange(sr) / sr
    tone = 0.5 * torch.sin(2 * torch.pi * 220 * t) * (1 + torch.sin(2 * torch.pi * 3 * t))
    silence = torch.zeros(sr)
    wav = torch.cat([silence, tone, silence]).unsqueeze(0)
    out = speech_only(wav)
    assert out.shape[1] <= wav.shape[1]
    assert speech_only(torch.zeros(1, sr)).shape[1] == sr   # no speech -> unchanged
    print(f"VAD OK ({wav.shape[1]} -> {out.shape[1]} samples)")
