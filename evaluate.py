import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """scores: fake probabilities (0-1), labels: 0=real, 1=fake. Returns EER in [0,1]."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    # Note: linear interpolation between the two ROC points straddling FPR==FNR —
    # this is the standard ASVspoof evaluation protocol; avoids picking an arbitrary threshold.
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))
    if diff[idx] == 0:
        return float(fpr[idx])
    # interpolate between adjacent crossing points
    if idx + 1 < len(diff) and np.sign(diff[idx]) != np.sign(diff[idx + 1]):
        f = interp1d([diff[idx], diff[idx + 1]], [fpr[idx], fpr[idx + 1]])
        return float(f(0.0))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Returns AUC-ROC score."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(auc(fpr, tpr))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    scores = rng.random(100)
    labels = rng.integers(0, 2, size=100)
    eer = compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0, f"EER out of range: {eer}"
    print("Evaluate OK")
