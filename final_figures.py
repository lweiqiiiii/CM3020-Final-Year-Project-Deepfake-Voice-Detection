"""Report figures + pilot cross-corpus scores for the final study.

  python3 final_figures.py spectrum            # fig_spectrum.png + spectrum_summary.json (600-file codec sample)
  python3 final_figures.py cross               # cross_corpus.json: AASIST / LCNN / fusion on data/real vs data/fake
  python3 final_figures.py bars [results.json] # fig_eer_conditions.png from results_final.json
  python3 final_figures.py attacks [results.json]  # fig_per_attack.png (A07-A19, eval_clean)
  python3 final_figures.py all                 # everything above
  python3 final_figures.py check               # self-check of the helper math on synthetic input

bars/attacks need `final_study.py report` first; cross needs the final
model_asvspoof_{clean,aug}.pth (written by `final_study.py lcnn`, seed 0);
model_asvspoof_vad.pth (`final_study.py lcnn_vad`) is optional and skipped if absent.
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
RESULTS = os.path.join(OUT, "results_final.json")
CONDS = ["clean", "g711", "g726", "mp3_low", "opus_low"]
LABELS = {"eval_clean": "eval", "clean": "Clean", "g711": "G.711", "g726": "G.726",
          "mp3_low": "MP3 32k", "opus_low": "Opus 16k", "vad": "VAD\n(clean)"}
# dataviz reference palette, categorical slots 1-5 in fixed order (same as plot_cv_results.py)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
VAD_LABEL = "LCNN VAD-trimmed + codec-aug (mean±sd)"
N_FFT, HOP, SR = 512, 256, 16000


def _style(ax):
    ax.yaxis.grid(True, color="#e5e5e2", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name, where=OUT):
    out = os.path.join(where, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── spectrum ──────────────────────────────────────────────────────────────────
def _logspec(wav):
    """1-D 16 kHz signal -> (N_FFT//2+1,) dB of the frame-averaged Hann power spectrum."""
    if len(wav) < N_FFT:
        wav = np.pad(wav, (0, N_FFT - len(wav)))
    idx = np.arange(0, len(wav) - N_FFT + 1, HOP)[:, None] + np.arange(N_FFT)
    power = np.abs(np.fft.rfft(wav[idx] * np.hanning(N_FFT), axis=1)) ** 2
    return 10 * np.log10(power.mean(axis=0) + 1e-12)


def _band_summary(freqs, diff, cut=4000):
    return {"mean_abs_diff_db_below_4k": float(np.abs(diff[freqs < cut]).mean()),
            "mean_abs_diff_db_above_4k": float(np.abs(diff[freqs > cut]).mean())}


def spectrum():
    from asvspoof import codec_sample
    from detect import _read
    from final_study import _variant
    sample = codec_sample()
    y = np.array([lab for _, lab, _ in sample])
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    diffs, summary = {}, {"n_bonafide": int((y == 0).sum()), "n_spoof": int((y == 1).sum()), "conditions": {}}
    with ThreadPoolExecutor(2) as ex:     # Note: 2 threads, the LCNN/AASIST jobs own the rest
        for c in CONDS:
            # Note: per-file dB, then mean over files, so loud files don't dominate the class mean
            S = np.array(list(ex.map(lambda e: _logspec(_read(_variant(e[0], c))[0].numpy()), sample)))
            diffs[c] = S[y == 1].mean(0) - S[y == 0].mean(0)
            hi = freqs > 4000   # absolute level shows whether there is any real energy left above 4 kHz
            summary["conditions"][c] = {**_band_summary(freqs, diffs[c]),
                                        "mean_power_db_above_4k_bonafide": float(S[y == 0][:, hi].mean()),
                                        "mean_power_db_above_4k_spoof": float(S[y == 1][:, hi].mean())}
            print(f"{c}: {summary['conditions'][c]}", flush=True)
    json.dump(summary, open(os.path.join(OUT, "spectrum_summary.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]   # secondary encoding beyond colour
    for c, col, ls in zip(CONDS, COLORS, styles):
        ax.plot(freqs / 1000, diffs[c], color=col, ls=ls, lw=1.8, label=LABELS[c], zorder=3)
    ax.axhline(0, color="#52514e", lw=0.8, zorder=2)
    ax.axvline(4, color="#52514e", lw=1, ls="--", zorder=2)
    ax.text(4.08, 0.97, "4 kHz: Nyquist of 8 kHz codecs", transform=ax.get_xaxis_transform(),
            va="top", fontsize=9, color="#52514e")
    # resampler roll-off is 3.8-4.7 kHz, so the 8 kHz-codec curves only break away near 4.7 kHz
    ax.text(4.9, 0.45, "G.711/G.726 above ~4.7 kHz: noise floor\n(~50 dB below clean), not signal",
            transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="#52514e")
    ax.set_xlim(0, 8)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Spoof − bona fide mean power (dB)")
    ax.set_title(f"Spoof vs bona fide spectral difference per codec "
                 f"(ASVspoof eval sample, n={len(sample)})", fontsize=11)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=9)
    _style(ax)
    _save(fig, "fig_spectrum.png")


# ── cross-corpus pilot ────────────────────────────────────────────────────────
def _metrics(scores, y):
    from evaluate import compute_eer, compute_auc
    s = np.asarray(scores)
    return {"eer": compute_eer(s, y), "auc": compute_auc(s, y), "mean_real": float(s[y == 0].mean()),
            "mean_fake": float(s[y == 1].mean()), "n": int(len(s))}


def _show(name, key, m):
    print(f"{name:<22} {key:<10} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n"), flush=True)


def _load_audio_tiled(path):
    """Speech-only, tiled to >= 4 s, then cut to 4 s — same input the VAD cache gave lcnn_vad."""
    from detect import _read, CLIP_SAMPLES
    from detect import tile
    return tile(_read(path, vad=True))[:, :CLIP_SAMPLES]


def cross():
    import torch
    from detect import LCNN, _extract, _load_audio
    from ensemble import aasist_input, aasist_prob
    torch.set_num_threads(2)
    files = ([(p, 0) for p in sorted(glob.glob(os.path.join(HERE, "data", "real", "*.wav")))]
             + [(p, 1) for p in sorted(glob.glob(os.path.join(HERE, "data", "fake", "*.wav")))])
    y = np.array([lab for _, lab in files])
    weights = {f"lcnn_{k}": os.path.join(HERE, "models", f"model_asvspoof_{k}.pth") for k in ("clean", "aug")}
    weights["lcnn_pilot_in_sample"] = os.path.join(HERE, "models", "model.pth")   # trained on these same 38 files
    vad_w = os.path.join(HERE, "models", "model_asvspoof_vad.pth")
    if os.path.exists(vad_w):
        weights["lcnn_vad"] = vad_w
    lcnn = {}
    for k, w in weights.items():
        lcnn[k] = LCNN()
        lcnn[k].load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        lcnn[k].eval()
    res = {"corpus": "data/real (label 0, own voice) vs data/fake (label 1, ElevenLabs clones)",
           "notes": {"lcnn_pilot_in_sample": "model.pth, trained on these same 38 files: in-sample, not a test",
                     "reverse_asvspoof_sample": "model.pth on the 600-file ASVspoof codec sample; "
                                                "mean_real = bona fide, mean_fake = spoof",
                     "lcnn_vad": ("VAD-trimmed + codec-aug LCNN; its 'vad' setting is speech-only then tile-padded "
                                  "to 4 s, as trained" if "lcnn_vad" in weights
                                  else "skipped: model_asvspoof_vad.pth not found (run final_study.py lcnn_vad)")},
           "lcnn_weights_mtime":{k: os.path.getmtime(w) for k, w in weights.items()}, "models": {}}
    for vad in (False, True):
        key = "vad" if vad else "no_vad"
        with torch.no_grad():
            sc = {"aasist": aasist_prob(torch.cat([aasist_input(p, vad) for p, _ in files])).tolist()}
            for k, m in lcnn.items():
                if k == "lcnn_vad" and vad:   # score it the way it was trained: tile, not zero-pad
                    sc[k] = [m(_extract(_load_audio_tiled(p))).item() for p, _ in files]
                else:
                    sc[k] = [m(_extract(_load_audio(p, vad))).item() for p, _ in files]
        sc["fusion_clean"] = [(a + b) / 2 for a, b in zip(sc["aasist"], sc["lcnn_clean"])]
        if "lcnn_vad" in sc:
            sc["fusion_vad"] = [(a + b) / 2 for a, b in zip(sc["aasist"], sc["lcnn_vad"])]
        for name, s in sc.items():
            res["models"].setdefault(name, {})[key] = _metrics(s, y)
            _show(name, key, res["models"][name][key])
    # reverse check: the pilot LCNN on the ASVspoof codec sample (mean_real = bona fide, mean_fake = spoof).
    # Note: AASIST skipped here — its clean/g711 numbers on this sample are in results_final.json
    from asvspoof import codec_sample
    from final_study import _variant
    sample = codec_sample()
    ys = np.array([lab for _, lab, _ in sample])
    res["reverse_asvspoof_sample"] = {"lcnn_pilot": {}}
    with torch.no_grad():
        for c in ("clean", "g711"):
            s = [lcnn["lcnn_pilot_in_sample"](_extract(_load_audio(_variant(p, c)))).item() for p, _, _ in sample]
            res["reverse_asvspoof_sample"]["lcnn_pilot"][c] = _metrics(s, ys)
            _show("lcnn_pilot", f"asv_{c}", res["reverse_asvspoof_sample"]["lcnn_pilot"][c])
    json.dump(res, open(os.path.join(OUT, "cross_corpus.json"), "w"), indent=1)
    print("wrote cross_corpus.json")


# ── results_final.json figures ────────────────────────────────────────────────
def _seed_stats(models, kind, get):
    """mean/std over lcnn_<kind>_s* of get(model_dict) -> arrays (works for scalars or aligned lists)."""
    vals = np.array([get(m) for n, m in sorted(models.items()) if n.startswith(f"lcnn_{kind}_s")])
    return vals.mean(0), vals.std(0)


def _grouped(ax, groups, series, width=0.2):
    """series = [(label, color, values%, err% or None)]; bars with value labels."""
    x = np.arange(len(groups))
    for i, (label, col, v, err) in enumerate(series):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * (width + 0.01), v, width, yerr=err, label=label,
                      color=col, zorder=3, error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"})
        tops = np.asarray(v) + (np.asarray(err) if err is not None else 0)   # label above the error bar
        for b, t in zip(bars, tops):
            ax.text(b.get_x() + b.get_width() / 2, t + 0.4, f"{b.get_height():.1f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90, color="#52514e")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("EER (%)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    _style(ax)


def bars(path=RESULTS):
    r = json.load(open(path))
    conds = ["eval_clean"] + CONDS + ["vad"]
    groups = [f"eval\n({r['n']['eval_clean']})"] + [LABELS[c] for c in conds[1:]]
    ss = r["seed_summary"]
    series = [(f"LCNN clean-trained (mean±sd, {ss['lcnn_clean']['clean']['n_seeds']} seeds)", COLORS[0],
               [ss["lcnn_clean"][c]["mean"] * 100 for c in conds], [ss["lcnn_clean"][c]["std"] * 100 for c in conds]),
              ("LCNN codec-augmented (mean±sd)", COLORS[1],
               [ss["lcnn_aug"][c]["mean"] * 100 for c in conds], [ss["lcnn_aug"][c]["std"] * 100 for c in conds]),
              ("AASIST", COLORS[2], [r["models"]["aasist"][c]["eer"] * 100 for c in conds], None),
              ("Fusion (AASIST + LCNN clean)", COLORS[3],
               [r["models"]["fusion_clean"][c]["eer"] * 100 for c in conds], None)]
    if "lcnn_vad" in ss:
        series.insert(2, (VAD_LABEL, COLORS[4], [ss["lcnn_vad"][c]["mean"] * 100 for c in conds],
                          [ss["lcnn_vad"][c]["std"] * 100 for c in conds]))
    fig, ax = plt.subplots(figsize=(12 if len(series) > 4 else 10, 5))
    _grouped(ax, groups, series, width=0.16 if len(series) > 4 else 0.2)
    ax.axvline(0.5, color="#52514e", lw=0.8, ls=":")
    top = max(max(v) + (max(e) if e else 0) for _, _, v, e in series)
    ax.set_ylim(0, top * 1.4)
    ax.set_title(f"EER by condition (ASVspoof 2019 LA, unseen attacks A07–A19)\n"
                 f"eval: full {r['n']['eval_clean']}-file set; codec and VAD groups: "
                 f"{r['n']['clean']}-file sample", fontsize=11)
    _save(fig, "fig_eer_conditions.png", os.path.dirname(os.path.abspath(path)))  # next to its data


def attacks(path=RESULTS):
    r = json.load(open(path))
    m = r["models"]
    att = sorted(m["aasist"]["per_attack_eer"])
    series = []
    for kind, label, col in (("clean", "LCNN clean-trained (mean±sd)", COLORS[0]),
                             ("aug", "LCNN codec-augmented (mean±sd)", COLORS[1]),
                             ("vad", VAD_LABEL, COLORS[4])):
        if not any(n.startswith(f"lcnn_{kind}_s") for n in m):
            continue
        mu, sd = _seed_stats(m, kind, lambda d: [d["per_attack_eer"][a] for a in att])
        series.append((label, col, mu * 100, sd * 100))
    series.append(("AASIST", COLORS[2], [m["aasist"]["per_attack_eer"][a] * 100 for a in att], None))
    fig, ax = plt.subplots(figsize=(13 if len(series) > 3 else 11, 5))
    _grouped(ax, att, series, width=0.2 if len(series) > 3 else 0.26)
    top = max(max(v) + (max(e) if e is not None else 0) for _, _, v, e in series)
    ax.set_ylim(0, top * 1.35)
    ax.set_xlabel("Eval attack (each vs all bona fide, eval set)")
    ax.set_title("Per-attack EER on the unseen-attack eval set", fontsize=11)
    _save(fig, "fig_per_attack.png", os.path.dirname(os.path.abspath(path)))


def check():
    sr = SR
    t = np.arange(sr) / sr
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    spec = _logspec(np.sin(2 * np.pi * 1000 * t))
    assert abs(f[spec.argmax()] - 1000) <= sr / N_FFT, f[spec.argmax()]
    assert _logspec(np.zeros(100)).shape == (N_FFT // 2 + 1,)          # short clip is padded
    d = np.where(f < 4000, 1.0, -3.0)
    s = _band_summary(f, d)
    assert s["mean_abs_diff_db_below_4k"] == 1.0 and s["mean_abs_diff_db_above_4k"] == 3.0
    models = {"lcnn_clean_s0": {"x": [0.1, 0.2]}, "lcnn_clean_s1": {"x": [0.3, 0.2]}, "aasist": {"x": [9, 9]}}
    mu, sd = _seed_stats(models, "clean", lambda d: d["x"])
    assert np.allclose(mu, [0.2, 0.2]) and np.allclose(sd, [0.1, 0.0])
    import torch
    from detect import tile
    assert tile(torch.ones(1, 3 * SR)).shape[1] >= 4 * SR       # 3 s speech -> tiled to >= 4 s
    assert tile(torch.ones(1, 9 * SR)).shape[1] == 9 * SR       # 9 s kept whole
    print("final_figures OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2:3]
    if cmd in ("bars", "attacks"):
        globals()[cmd](*arg)
    elif cmd in ("spectrum", "cross", "check"):
        globals()[cmd]()
    elif cmd == "all":
        spectrum(); cross(); bars(); attacks()
    else:
        sys.exit(__doc__)
