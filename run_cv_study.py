"""Cross-validated codec robustness study.

Fixes the validity flaw in run_study.py: there, the model scored its own
training data, so EER/AUC were saturated at 0/1. Here every file is scored
by a model that never saw it (stratified 5-fold CV), giving honest
generalization estimates from the same 38 files.

Per fold, two models are trained on identical file splits:
  - clean-trained:  original .wav files only (the baseline detector)
  - codec-trained:  clean + all four codec-degraded versions of each
                    training file (codec-aware augmentation)
Both score the held-out files under all five conditions. Comparing the two
answers the mitigation question: does codec-aware training close the
robustness gap that phone codecs open?

Output: results_cv.json (pooled EER/AUC/mean scores per condition per
variant, plus raw per-file scores). Chart is plotted separately by
plot_cv_results.py so the study never reruns just to restyle a figure.

Multi-window check (detect.score_windows, 4 s windows / 2 s hop, mean prob):
  python3 run_cv_study.py --multiwindow   same folds/seeds/models, every held-out
      file scored both single-window (first 4 s) and multi-window
      -> results_multiwindow.json["pilot_cv"]; results_cv.json untouched
  python3 run_cv_study.py --asvspoof-multiwindow   model_asvspoof_clean.pth on the
      2394-file ASVspoof eval_clean set, single vs multi
      -> results_multiwindow.json["asvspoof_eval_clean"]

Leading-silence confound (pilot real clips open with ~2 s silence, fakes don't):
  python3 run_cv_study.py --vad   silence stats + seeded CV default vs VAD-trimmed/tiled
      inputs + candidate model_pilot_vad.pth -> results_pilot_vad.json
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

from detect import _train_model, _extract, _load_audio, _read, _window_starts, score_windows, tile, CLIP_SAMPLES, LCNN
from degrade import degrade_all, cleanup
from evaluate import compute_eer, compute_auc

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = ["clean", "g711", "g726", "mp3_low", "opus_low"]
VARIANTS = ["clean_trained", "codec_trained"]
K = 5
EPOCHS = 20
SEED = 0
MW_OUT = os.path.join(HERE, "results", "results_multiwindow.json")


def _score(model, path):
    with torch.no_grad():
        return model(_extract(_load_audio(path))).item()


def _score_multi(model, path):
    return score_windows(model, _read(path))[0]


def _metrics(s, y):
    return {"eer": compute_eer(s, y), "auc": compute_auc(s, y),
            "mean_score_real": float(s[y == 0].mean()),
            "mean_score_fake": float(s[y == 1].mean())}


def _save_mw(key, value):
    """Read-modify-write one section of results_multiwindow.json."""
    out = json.load(open(MW_OUT)) if os.path.exists(MW_OUT) else {}
    out[key] = value
    with open(MW_OUT, "w") as f:
        json.dump(out, f, indent=2)


def _folds():
    real = sorted(glob.glob(os.path.join(HERE, "data/real/*.wav")))
    fake = sorted(glob.glob(os.path.join(HERE, "data/fake/*.wav")))
    rng = random.Random(SEED)
    rng.shuffle(real)
    rng.shuffle(fake)
    # stratified folds: every k-th file of each class
    folds = [[(f, 0) for f in real[i::K]] + [(f, 1) for f in fake[i::K]]
             for i in range(K)]
    print(f"{len(real) + len(fake)} files ({len(real)} real, {len(fake)} fake), {K} folds")
    return real, fake, folds


def _cv(folds, src, scorers, t0):
    """K-fold CV; src(path, condition) -> file actually trained on / scored.
    Returns scores[mode][variant][condition][path] from held-out models.
    torch is seeded (init + dropout) per fold, so runs are reproducible; results_cv.json
    predates this (unseeded) and was not re-run."""
    all_scores = {m: {v: {c: {} for c in CONDITIONS} for v in VARIANTS} for m in scorers}
    for k, held in enumerate(folds):
        train_pairs = [x for j, fold in enumerate(folds) if j != k for x in fold]
        pairs = {"clean_trained": [(src(p, "clean"), y) for p, y in train_pairs],
                 "codec_trained": [(src(p, c), y) for p, y in train_pairs for c in CONDITIONS]}
        print(f"\nfold {k + 1}/{K}: {len(pairs['clean_trained'])} clean / "
              f"{len(pairs['codec_trained'])} augmented train files, {len(held)} held out")
        models = {}
        for v in VARIANTS:
            torch.manual_seed(SEED + k)
            models[v], *_ = _train_model(pairs[v], epochs=EPOCHS,
                                         rng=random.Random(SEED + k), verbose=False)
            models[v].eval()
            for p, y in held:
                for c in CONDITIONS:
                    for m, fn in scorers.items():
                        all_scores[m][v][c][p] = fn(models[v], src(p, c))
        print(f"fold {k + 1} done ({time.time() - t0:.0f}s elapsed)")
    return all_scores


def main(multiwindow=False):
    scorers = {"single": _score, "multi": _score_multi} if multiwindow else {"single": _score}
    t0 = time.time()
    real, fake, folds = _folds()
    all_files = [x for fold in folds for x in fold]

    print("degrading all files once up front...")
    degraded = {p: degrade_all(p) for p, _ in all_files}
    try:   # degraded[p]["clean"] is p itself
        all_scores = _cv(folds, lambda p, c: degraded[p][c], scorers, t0)
    finally:
        for d in degraded.values():
            cleanup(d)
    scores = all_scores["single"]

    labels = {p: y for p, y in all_files}
    order = [p for p, _ in all_files]
    y = np.array([labels[p] for p in order])

    if multiwindow:
        n_win = {os.path.relpath(p, HERE): len(_window_starts(_read(p).shape[1])) for p in order}
        _save_mw("pilot_cv", {
            "method": f"stratified {K}-fold CV, {EPOCHS} epochs, seed {SEED}; same models score "
                      "each held-out file single-window (first 4 s) and multi-window "
                      "(mean of 4 s windows, 2 s hop, tail window at clip end)",
            "n_real": len(real), "n_fake": len(fake),
            "n_windows": n_win,
            "per_condition": {m: {v: {c: _metrics(np.array([all_scores[m][v][c][p] for p in order]), y)
                                      for c in CONDITIONS} for v in VARIANTS} for m in scorers},
            "raw_scores": {m: {v: {c: {os.path.relpath(p, HERE): all_scores[m][v][c][p] for p in order}
                                   for c in CONDITIONS} for v in VARIANTS} for m in scorers},
            "runtime_s": round(time.time() - t0),
        })
        for v in VARIANTS:
            print(f"\n== {v}: single vs multi ==")
            print(f"{'condition':<10} {'EER s/m':>13} {'AUC s/m':>13} {'real s/m':>13} {'fake s/m':>13}")
            for c in CONDITIONS:
                r = [_metrics(np.array([all_scores[m][v][c][p] for p in order]), y) for m in scorers]
                print(f"{c:<10} " + " ".join(f"{r[0][k]:>6.3f}/{r[1][k]:<6.3f}" for k in
                      ("eer", "auc", "mean_score_real", "mean_score_fake")))
        print(f"\ntotal {time.time() - t0:.0f}s — wrote {MW_OUT} (results_cv.json untouched)")
        return

    results = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for c in CONDITIONS:
            s = np.array([scores[v][c][p] for p in order])
            results[v][c] = _metrics(s, y)

    out = {
        "method": f"stratified {K}-fold CV, {EPOCHS} epochs, seed {SEED}; "
                  "every score comes from a model that never saw the file",
        "n_real": len(real), "n_fake": len(fake),
        "per_condition": results,
        "raw_scores": {v: {c: {os.path.relpath(p, HERE): scores[v][c][p]
                               for p in order} for c in CONDITIONS} for v in VARIANTS},
    }
    with open(os.path.join(HERE, "results", "results_cv.json"), "w") as f:
        json.dump(out, f, indent=2)

    for v in VARIANTS:
        print(f"\n== {v} ==")
        print(f"{'condition':<10} {'EER':>6} {'AUC':>6} {'mean(real)':>11} {'mean(fake)':>11}")
        for c in CONDITIONS:
            r = results[v][c]
            print(f"{c:<10} {r['eer']:>6.3f} {r['auc']:>6.3f} "
                  f"{r['mean_score_real']:>11.3f} {r['mean_score_fake']:>11.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s — wrote results_cv.json")


PV_OUT = os.path.join(HERE, "results", "results_pilot_vad.json")


def _metrics05(s, y):
    return {**_metrics(s, y), "false_alarms": int((s[y == 0] > 0.5).sum()),
            "misses": int((s[y == 1] <= 0.5).sum())}


def _silence_stats(files):
    """Leading silence / length / speech seconds per file from Silero timestamps."""
    import vad
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad._model = vad._model or load_silero_vad()
    out = {}
    for p, y in files:
        wav = _read(p)
        ts = get_speech_timestamps(wav[0], vad._model, sampling_rate=16000)
        out[os.path.relpath(p, HERE)] = {
            "label": y, "length_s": wav.shape[1] / 16000,
            "leading_silence_s": ts[0]["start"] / 16000 if ts else wav.shape[1] / 16000,
            "speech_s": sum(t["end"] - t["start"] for t in ts) / 16000}
    summary = {}
    for y, name in ((0, "real"), (1, "fake")):
        summary[name] = {k: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                         for k in ("leading_silence_s", "length_s", "speech_s")
                         for v in [[d[k] for d in out.values() if d["label"] == y]]}
    lab = np.array([d["label"] for d in out.values()])
    lead = np.array([d["leading_silence_s"] for d in out.values()])
    summary["eer_of_minus_leading_silence"] = compute_eer(-lead, lab)
    summary["auc_of_minus_leading_silence"] = compute_auc(-lead, lab)
    return summary, out


def pilot_vad():
    """Leading-silence confound check + seeded CV default vs VAD-trimmed/tiled + candidate model."""
    import tempfile, shutil
    import soundfile as sf
    from vad import speech_only
    t0 = time.time()
    real, fake, folds = _folds()
    all_files = [x for fold in folds for x in fold]
    order = [p for p, _ in all_files]
    y = np.array([lab for _, lab in all_files])
    summary, per_file = _silence_stats(all_files)
    print(json.dumps(summary, indent=1))

    degraded = {p: degrade_all(p) for p in order}
    tmp = tempfile.mkdtemp(prefix="pilot_vad_")
    try:
        # VAD after the codec round-trip (as on a real call), then tile to >= 4 s
        vadded = {}
        for i, p in enumerate(order):
            for c in CONDITIONS:
                dst = os.path.join(tmp, f"{i}_{c}.wav")
                sf.write(dst, tile(speech_only(_read(degraded[p][c])))[0].numpy(), 16000)
                vadded[p, c] = dst
        print(f"VAD cache built ({time.time() - t0:.0f}s)")
        srcs = {"default": lambda p, c: degraded[p][c], "vad": lambda p, c: vadded[p, c]}
        res = {}
        for mode, src in srcs.items():
            print(f"\n=== {mode} inputs ===")
            sc = _cv(folds, src, {"single": _score}, t0)["single"]
            res[mode] = {"per_condition": {v: {c: _metrics05(np.array([sc[v][c][p] for p in order]), y)
                                               for c in CONDITIONS} for v in VARIANTS},
                         "raw_scores": {v: {c: {os.path.relpath(p, HERE): sc[v][c][p] for p in order}
                                            for c in CONDITIONS} for v in VARIANTS}}
        # candidate app model: all 38 clips, VAD+tile, clean + 4 codec versions (= codec_trained)
        torch.manual_seed(SEED)
        model, loss, acc = _train_model([(vadded[p, c], lab) for p, lab in all_files for c in CONDITIONS],
                                        epochs=EPOCHS, rng=random.Random(SEED), verbose=False)
        torch.save(model.state_dict(), os.path.join(HERE, "models", "model_pilot_vad.pth"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for d in degraded.values():
            cleanup(d)

    out = {
        "method": f"stratified {K}-fold CV (same folds as results_cv.json), {EPOCHS} epochs, "
                  f"torch.manual_seed(SEED + fold) before each model (seeded, unlike results_cv.json). "
                  "default = untrimmed first 4 s zero-padded (as results_cv.json); vad = codec round-trip -> "
                  "Silero speech_only -> tile-repeat to >= 4 s -> first 4 s, for train and score. "
                  "false_alarms / misses counted at threshold 0.5.",
        "leading_silence": {"summary": summary, "per_file": per_file},
        "cv": res,
        "candidate_model": {"path": "model_pilot_vad.pth", "train": "all 38 clips x 5 conditions, VAD+tile, "
                            f"{EPOCHS} epochs, seed {SEED}", "final_loss": round(loss, 4), "train_acc": round(acc, 4),
                            "note": "expects VAD-trimmed + tiled input (vad.speech_only then tile), not the raw clip"},
        "runtime_s": round(time.time() - t0),
    }
    with open(PV_OUT, "w") as f:
        json.dump(out, f, indent=2)
    for v in VARIANTS:
        print(f"\n== {v}: default / vad ==")
        for c in CONDITIONS:
            a, b = (res[m]["per_condition"][v][c] for m in ("default", "vad"))
            print(f"{c:<9} " + "  ".join(f"{k.replace('mean_score_', '')} {a[k]:.3f}/{b[k]:.3f}" if isinstance(a[k], float)
                                         else f"{k} {a[k]}/{b[k]}" for k in a))
    print(f"\ntotal {time.time() - t0:.0f}s — wrote {PV_OUT}, model_pilot_vad.pth")


def asvspoof_multiwindow():
    """Seed-0 clean ASVspoof LCNN on eval_clean: single-window vs multi-window."""
    from asvspoof import _pairs
    from collections import Counter
    t0 = time.time()
    model = LCNN()
    model.load_state_dict(torch.load(os.path.join(HERE, "models", "model_asvspoof_clean.pth"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    entries = _pairs("eval")
    single, multi, n_win = [], [], []
    for i, (p, _, _) in enumerate(entries):
        wav = _read(p)
        single.append(score_windows(model, wav[:, :CLIP_SAMPLES])[0])   # == _load_audio pad/trim
        prob, n = score_windows(model, wav)
        multi.append(prob)
        n_win.append(n)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(entries)}  {time.time() - t0:.0f}s", flush=True)
    y = np.array([lab for _, lab, _ in entries])
    single, multi = np.array(single), np.array(multi)
    res = {
        "model": "model_asvspoof_clean.pth (seed-0 clean benchmark LCNN)",
        "n": len(entries),
        "single": _metrics(single, y), "multi": _metrics(multi, y),
        "n_windows_hist": {str(k): v for k, v in sorted(Counter(n_win).items())},
        "n_files_changed": int((np.abs(single - multi) > 1e-6).sum()),
        "runtime_s": round(time.time() - t0),
    }
    ref = os.path.join(HERE, "results", "scores_final", "lcnn_clean_s0.json")
    if os.path.exists(ref) and len(json.load(open(ref))["eval_clean"]) == len(single):   # sanity: single-window must reproduce final_study's seed-0 scores
        res["single_max_abs_diff_vs_scores_final"] = float(
            np.abs(single - np.array(json.load(open(ref))["eval_clean"])).max())
    _save_mw("asvspoof_eval_clean", res)
    print(json.dumps({k: v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    if "--asvspoof-multiwindow" in sys.argv:
        asvspoof_multiwindow()
    elif "--vad" in sys.argv:
        pilot_vad()
    else:
        main(multiwindow="--multiwindow" in sys.argv)
