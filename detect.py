"""
deepfake voice detection — LCNN with MFM activation on LFCC features.
Note: single conv→pool block instead of full LCNN-29 depth; enough for
          proof-of-concept without the full ASVspoof competition stack.
"""

import os
import glob
import random
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CLIP_SAMPLES  = 64_000          # 4 s × 16 kHz
N_LFCC        = 20
N_FILTER      = 20
N_FFT         = 512
HOP_LENGTH    = 160             # 10 ms hop → ~400 frames for 4 s clip
WINDOW_HOP    = 32_000          # 2 s hop between 4 s scoring windows (score_windows)
MIN_TAIL      = 8_000           # 0.5 s: skip a tail window this close to the last regular one


# ── feature extraction ────────────────────────────────────────────────────────
_lfcc_transform = T.LFCC(
    sample_rate=SAMPLE_RATE,
    n_filter=N_FILTER,
    n_lfcc=N_LFCC,
    speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "center": False},
)


def _read(path: str, vad: bool = False) -> torch.Tensor:
    """Load full clip as 16 kHz mono (1, N); vad=True keeps only Silero-detected speech."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    # NaN/inf samples (a corrupt or crafted float WAV) would make every score NaN
    wav = torch.nan_to_num(torch.from_numpy(data.T), nan=0.0, posinf=0.0, neginf=0.0)   # (channels, samples)
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav.mean(dim=0, keepdim=True)          # stereo → mono
    if vad:
        from vad import speech_only
        wav = speech_only(wav)
    return wav


def tile(wav: torch.Tensor, n: int = CLIP_SAMPLES) -> torch.Tensor:
    """(1, N) -> (1, >=n): repeat short clips instead of zero-padding.
    Note: after VAD bona fide clips shrink more than spoofs, so _load_audio's zero-pad
    length would become a new class cue; tiling (as AASIST does) leaves no silence to count."""
    if wav.shape[1] >= n:
        return wav
    return wav.repeat(1, -(-n // wav.shape[1]))[:, :n]


def _load_audio(path: str, vad: bool = False) -> torch.Tensor:
    """Load, resample to 16 kHz, mono, pad/trim to 4 s."""
    wav = _read(path, vad)
    if wav.shape[1] < CLIP_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, CLIP_SAMPLES - wav.shape[1]))
    else:
        wav = wav[:, :CLIP_SAMPLES]
    return wav                                   # (1, 64000)


def _extract(wav: torch.Tensor) -> torch.Tensor:
    """Return LFCC tensor shaped (1, 1, n_lfcc, time_frames)."""
    feats = _lfcc_transform(wav)                 # (1, n_lfcc, T)
    feats = (feats - feats.mean()) / (feats.std() + 1e-9)  # unit normalize — prevents gradient explosion
    return feats.unsqueeze(0)                    # (1, 1, n_lfcc, T)


def _window_starts(n: int) -> list:
    """Sample offsets of the 4 s windows score_windows uses for an n-sample clip."""
    if n <= CLIP_SAMPLES:
        return [0]
    starts = list(range(0, n - CLIP_SAMPLES + 1, WINDOW_HOP))
    if n - CLIP_SAMPLES - starts[-1] >= MIN_TAIL:
        starts.append(n - CLIP_SAMPLES)          # tail window ending at the clip end
    return starts


def score_windows(model: nn.Module, wav: torch.Tensor) -> tuple:
    """Mean fake probability over 4 s windows of a full (1, N) 16 kHz clip.

    Windows start every WINDOW_HOP (2 s). Tail rule: if the last regular window
    ends >= 0.5 s (MIN_TAIL) before the clip does, one extra window is added
    ending exactly at the clip end (it overlaps its predecessor); a shorter
    leftover (< 0.5 s) is left unscored rather than adding a near-duplicate window.
    A clip <= 4 s gives one window zero-padded like _load_audio, so short clips
    score identically to single-window scoring. E.g. 3 s -> 1, 4 s -> 1,
    4.3 s -> 1 (0.3 s leftover), 6 s -> 2 (0-4, 2-6), 9 s -> 4 (0-4, 2-6, 4-8, tail 5-9).
    Each window is normalised by _extract on its own; one batched forward pass.
    Caller sets model.eval(). Returns (mean_prob, n_windows).
    """
    starts = _window_starts(wav.shape[1])
    wav = torch.nn.functional.pad(wav, (0, max(0, CLIP_SAMPLES - wav.shape[1])))
    feats = torch.cat([_extract(wav[:, s:s + CLIP_SAMPLES]) for s in starts])
    with torch.no_grad():
        probs = model(feats)
    return probs.mean().item(), len(starts)


# ── model ─────────────────────────────────────────────────────────────────────
class MFM(nn.Module):
    """Max Feature Map — splits channels in half, keeps element-wise max."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    def __init__(self, n_lfcc: int = N_LFCC):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),   # → 32ch
            MFM(),                                         # → 16ch
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),  # → 64ch
            MFM(),                                         # → 32ch
            nn.MaxPool2d(2, 2),
        )
        # Note: derive T from actual LFCC params so the Linear input size is exact
        with torch.no_grad():
            actual_T = (CLIP_SAMPLES - N_FFT) // HOP_LENGTH + 1  # 397 with defaults
            dummy = torch.zeros(1, 1, n_lfcc, actual_T)
            flat = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1)).squeeze(1)


# ── public API ────────────────────────────────────────────────────────────────
def predict(audio_path: str, model_path: str = "models/model.pth", multiwindow: bool = False) -> dict:
    """
    Classify a single audio file.
    Scores the first 4 s (as the app does); multiwindow=True scores
    the full clip with score_windows (mean over 4 s windows, 2 s hop).
    Returns {"fake_probability": float, "verdict": str, "features_shape": str,
             "n_windows": int}  (features_shape is per window)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Run train() first or provide a valid model_path."
        )
    model = LCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    wav = _read(audio_path)
    prob, n_windows = score_windows(model, wav if multiwindow else wav[:, :CLIP_SAMPLES])

    return {
        "fake_probability": round(prob, 4),
        "verdict": "FAKE" if prob > 0.5 else "REAL",
        "features_shape": str((1, 1, N_LFCC, (CLIP_SAMPLES - N_FFT) // HOP_LENGTH + 1)),
        "n_windows": n_windows,
    }


def _train_model(files: list, epochs: int = 10, lr: float = 1e-4,
                 rng: random.Random = None, verbose: bool = True):
    """Train an LCNN on a list of (path, label) pairs.
    Returns (model, final_loss, train_accuracy)."""
    rng = rng or random.Random(0)
    model     = LCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    final_loss, correct = 0.0, 0
    for epoch in range(epochs):
        rng.shuffle(files)          # reshuffle so classes interleave each epoch
        epoch_loss, epoch_correct = 0.0, 0
        for path, label in files:
            try:
                wav   = _load_audio(path)
                feats = _extract(wav)
            except Exception:
                continue
            target = torch.tensor([float(label)])
            optimizer.zero_grad()
            pred = model(feats)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss    += loss.item()
            epoch_correct += int((pred.item() > 0.5) == bool(label))
        final_loss = epoch_loss / max(len(files), 1)
        correct    = epoch_correct
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}  "
                  f"acc={correct / max(len(files), 1):.3f}")
    return model, final_loss, correct / max(len(files), 1)


def train(
    real_dir: str,
    fake_dir: str,
    epochs: int = 10,
    save_path: str = "models/model.pth",
) -> dict:
    """
    Train LCNN on .wav files from real_dir (label 0) and fake_dir (label 1).
    Holds out 20% of each class for validation.
    Returns {"final_loss": float, "accuracy": float, "val_accuracy": float}
    """
    real = sorted(glob.glob(os.path.join(real_dir, "*.wav")))
    fake = sorted(glob.glob(os.path.join(fake_dir, "*.wav")))
    if not real or not fake:
        raise ValueError(f"No .wav files found in '{real_dir}' or '{fake_dir}'.")

    # stratified 80/20 split, seeded so runs are reproducible
    rng = random.Random(0)
    rng.shuffle(real)
    rng.shuffle(fake)
    n_r, n_f = max(1, len(real) // 5), max(1, len(fake) // 5)
    val_files = [(f, 0) for f in real[:n_r]] + [(f, 1) for f in fake[:n_f]]
    files     = [(f, 0) for f in real[n_r:]] + [(f, 1) for f in fake[n_f:]]

    model, final_loss, accuracy = _train_model(files, epochs=epochs, rng=rng)

    # held-out validation accuracy — dropout off, no gradients
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for path, label in val_files:
            prob = model(_extract(_load_audio(path))).item()
            val_correct += int((prob > 0.5) == bool(label))
    val_accuracy = val_correct / max(len(val_files), 1)
    print(f"Validation acc={val_accuracy:.3f}  ({val_correct}/{len(val_files)} held-out files)")

    torch.save(model.state_dict(), save_path)
    return {
        "final_loss": round(final_loss, 4),
        "accuracy": round(accuracy, 4),
        "val_accuracy": round(val_accuracy, 4),
    }


# ── self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = LCNN().eval()
    with torch.no_grad():
        p = m(_extract(torch.zeros(1, CLIP_SAMPLES))).item()
    assert 0.0 <= p <= 1.0, f"forward pass returned {p}"
    # multi-window: count per the tail rule in score_windows; short clip == single-window score
    wav9 = torch.randn(1, 9 * SAMPLE_RATE) * 0.1
    for secs, want in [(3, 1), (4, 1), (4.3, 1), (4.6, 2), (6, 2), (9, 4)]:
        prob, n = score_windows(m, wav9[:, :int(secs * SAMPLE_RATE)])
        assert n == want, f"{secs}s -> {n} windows, expected {want}"
        assert 0.0 <= prob <= 1.0, f"{secs}s prob {prob}"
    short = wav9[:, :3 * SAMPLE_RATE]
    with torch.no_grad():
        single = m(_extract(torch.nn.functional.pad(short, (0, CLIP_SAMPLES - short.shape[1])))).item()
    assert abs(score_windows(m, short)[0] - single) < 1e-6, "short clip must match single-window"
    print("Multi-window OK (3s->1, 4s->1, 4.3s->1, 4.6s->2, 6s->2, 9s->4)")
    print("Pipeline OK")
