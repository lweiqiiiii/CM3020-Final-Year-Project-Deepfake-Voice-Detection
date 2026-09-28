"""Pre-trained AASIST (model #3) + score fusion with the bespoke LCNN.

AASIST weights/code: github.com/clovaai/aasist (MIT), vendored in aasist/.
Weights are committed; to re-fetch, see the AASIST repository above.
"""
import json
import os
import sys

import torch

from detect import _read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "aasist"))
AASIST_LEN = 64600          # ~4.04 s, the length AASIST was trained on

_model = None


def load_aasist():
    global _model
    if _model is None:
        from AASIST import Model
        cfg = json.load(open(os.path.join(HERE, "aasist", "AASIST.conf")))["model_config"]
        _model = Model(cfg)
        _model.load_state_dict(torch.load(os.path.join(HERE, "aasist", "AASIST.pth"),
                                          map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def aasist_input(path: str, vad: bool = False) -> torch.Tensor:
    """(1, 64600) raw waveform, short clips tile-padded exactly as the AASIST repo does."""
    wav = _read(path, vad)[0]
    reps = AASIST_LEN // wav.shape[0] + 1
    return wav.repeat(reps)[:AASIST_LEN].unsqueeze(0)


def aasist_prob(batch: torch.Tensor) -> torch.Tensor:
    """(B, 64600) -> (B,) probability of spoof. AASIST logits are [spoof, bonafide]."""
    with torch.no_grad():
        _, logits = load_aasist()(batch)
    return torch.softmax(logits, dim=1)[:, 0]


def fuse(p_lcnn: float, p_aasist: float) -> float:
    # Note: unweighted mean of probabilities; logistic-regression fusion
    # needs a held-out calibration set we don't have beyond eval.
    return (p_lcnn + p_aasist) / 2


if __name__ == "__main__":
    p = aasist_prob(torch.zeros(2, AASIST_LEN))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()
    assert fuse(0.2, 0.6) == 0.4
    print("Ensemble OK", p.tolist())
