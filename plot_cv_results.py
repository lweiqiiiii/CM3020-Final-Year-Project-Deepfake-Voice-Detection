"""Chart the cross-validated codec study: drop in mean fake score vs the
clean condition, clean-trained vs codec-trained. Reads results_cv.json
(run run_cv_study.py first); writes results_cv.png.

EER is 0.000 under every condition (in the JSON/report table), so an EER
chart would be five empty bars — the informative signal is score drift
toward the decision boundary, which is what this plots. Kept separate from
the study so restyling never reruns the 10-model training."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODECS = ["g711", "g726", "mp3_low", "opus_low"]
LABELS = ["G.711", "G.726", "MP3 32k", "Opus 16k"]
# dataviz reference palette slots 1–2, validated (CVD dE 73.6; aqua's contrast
# WARN is relieved by the value label on every bar)
SERIES = [("clean_trained", "Clean-trained", "#2a78d6"),
          ("codec_trained", "Codec-trained", "#1baf7a")]


def main():
    with open(os.path.join(HERE, "results", "results_cv.json")) as f:
        res = json.load(f)["per_condition"]

    x = np.arange(len(CODECS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (key, label, color) in enumerate(SERIES):
        clean = res[key]["clean"]["mean_score_fake"]
        drops = [(clean - res[key][c]["mean_score_fake"]) * 100 for c in CODECS]
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), drops, width,
                      label=label, color=color, zorder=3)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Drop in mean fake score vs clean (points)")
    ax.set_title("Codec-induced score drift toward the decision boundary\n"
                 "(stratified 5-fold CV; EER stayed 0.000 in all conditions — "
                 "drift is the early-warning signal)",
                 fontsize=11)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#e5e5e2", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    all_drops = [(res[k]["clean"]["mean_score_fake"] - res[k][c]["mean_score_fake"]) * 100
                 for k, _, _ in SERIES for c in CODECS]
    ax.set_ylim(min(0, min(all_drops) * 1.25), max(all_drops) * 1.25)
    plt.tight_layout()
    out = os.path.join(HERE, "results", "results_cv.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
