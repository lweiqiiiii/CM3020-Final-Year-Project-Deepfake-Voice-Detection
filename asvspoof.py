"""ASVspoof 2019 LA pipeline — benchmark-credible EER for the detector.

Dataset: https://datashare.ed.ac.uk/handle/10283/3336 (LA.zip, 7.12 GB)
expected at ../datasets/asvspoof2019/LA.zip relative to this repo.

CPU budget forces subsampling: the full LA train partition is 25,380
utterances (~days of CPU training). prep extracts a seeded, attack-stratified
subset directly from the zip (never fully extracted, ~19 GB disk saved):

  train: 1290 bonafide + 1290 spoof (215 per attack A01-A06), balanced
  eval:  600 bonafide + 1794 spoof (138 per attack A07-A19)

Eval attacks are disjoint from train attacks by ASVspoof design, so the
resulting EER is a genuine unseen-attack generalization number.

Usage (run in order; each step is resumable/idempotent):
  python3 asvspoof.py prep            # extract protocols + sampled flacs
  python3 asvspoof.py train [epochs]  # -> model_asvspoof.pth (~45 min CPU)
  python3 asvspoof.py eval            # clean EER/AUC (+ per-attack) -> results_asvspoof.json
  python3 asvspoof.py codecs          # codec study on 600-file eval subsample
  python3 asvspoof.py all             # train + eval + codecs

Smoke test: python3 asvspoof.py smoke  (tiny caps, minutes, exercises every step)
"""
import json
import os
import random
import sys
import zipfile

import numpy as np
import torch

from detect import _train_model, _extract, _load_audio, LCNN
from degrade import degrade_all, cleanup
from evaluate import compute_eer, compute_auc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "datasets", "asvspoof2019")
ZIP = os.path.join(DATA, "LA.zip")
SUBSET = os.path.join(DATA, "LA_subset")
MANIFEST = os.path.join(SUBSET, "manifest.json")
MODEL = os.path.join(HERE, "models", "model_asvspoof.pth")
RESULTS = os.path.join(HERE, "results", "results_asvspoof.json")
SEED = 0

TRAIN_BONA, TRAIN_PER_ATTACK = 1290, 215   # 6 train attacks -> 1290 spoof
EVAL_BONA, EVAL_PER_ATTACK = 600, 138      # 13 eval attacks -> 1794 spoof
CODEC_BONA, CODEC_SPOOF = 150, 450
CONDITIONS = ["clean", "g711", "g726", "mp3_low", "opus_low"]


def _read_protocol(zf: zipfile.ZipFile, name: str):
    """Yield (trial_id, attack, label) from a cm protocol file inside the zip."""
    with zf.open(f"LA/ASVspoof2019_LA_cm_protocols/{name}") as f:
        for line in f.read().decode().splitlines():
            parts = line.split()          # speaker trial - attack key
            yield parts[1], parts[3], 0 if parts[4] == "bonafide" else 1


def _sample(entries, n_bona, per_attack, rng):
    """Seeded balanced sample: n_bona bonafide + per_attack of each attack."""
    bona = [e for e in entries if e[2] == 0]
    picked = rng.sample(bona, min(n_bona, len(bona)))
    attacks = sorted({e[1] for e in entries if e[2] == 1})
    for a in attacks:
        pool = [e for e in entries if e[1] == a]
        picked += rng.sample(pool, min(per_attack, len(pool)))
    return picked


def prep(limit=None):
    if not os.path.exists(ZIP):
        sys.exit(f"LA.zip not found at {ZIP} — download still running?")
    rng = random.Random(SEED)
    manifest = {"train": [], "eval": []}
    with zipfile.ZipFile(ZIP) as zf:
        picks = {
            "train": ("ASVspoof2019_LA_train",
                      _sample(list(_read_protocol(zf, "ASVspoof2019.LA.cm.train.trn.txt")),
                              TRAIN_BONA, TRAIN_PER_ATTACK, rng)),
            "eval": ("ASVspoof2019_LA_eval",
                     _sample(list(_read_protocol(zf, "ASVspoof2019.LA.cm.eval.trl.txt")),
                             EVAL_BONA, EVAL_PER_ATTACK, rng)),
        }
        for part, (zdir, entries) in picks.items():
            if limit:
                entries = entries[:limit // 2] + entries[-limit // 2:]  # keep both classes
            out_dir = os.path.join(SUBSET, part)
            os.makedirs(out_dir, exist_ok=True)
            for i, (trial, attack, label) in enumerate(entries):
                dest = os.path.join(out_dir, f"{trial}.flac")
                if not os.path.exists(dest):      # idempotent / resumable
                    with zf.open(f"LA/{zdir}/flac/{trial}.flac") as src, open(dest, "wb") as out:
                        out.write(src.read())
                manifest[part].append({"path": os.path.relpath(dest, HERE),
                                       "attack": attack, "label": label})
                if (i + 1) % 500 == 0:
                    print(f"{part}: {i + 1}/{len(entries)}")
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    for part in manifest:
        n = len(manifest[part])
        nb = sum(1 for e in manifest[part] if e["label"] == 0)
        print(f"{part}: {n} files ({nb} bonafide, {n - nb} spoof)")


def _pairs(part):
    with open(MANIFEST) as f:
        return [(os.path.join(HERE, e["path"]), e["label"], e["attack"])
                for e in json.load(f)[part]]


def train(epochs=10, limit=None):
    files = [(p, y) for p, y, _ in _pairs("train")][:limit]
    print(f"training on {len(files)} files, {epochs} epochs")
    model, loss, acc = _train_model(files, epochs=epochs, rng=random.Random(SEED))
    torch.save(model.state_dict(), MODEL)
    print(f"saved {MODEL}  final_loss={loss:.4f}  train_acc={acc:.3f}")


def _load_model():
    m = LCNN()
    m.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    return m.eval()


def _score_files(model, paths):
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            out.append(model(_extract(_load_audio(p))).item())
            if (i + 1) % 500 == 0:
                print(f"scored {i + 1}/{len(paths)}")
    return np.array(out)


def evaluate(limit=None):
    entries = _pairs("eval")
    if limit:
        entries = entries[:limit // 2] + entries[-limit // 2:]  # keep both classes
    model = _load_model()
    scores = _score_files(model, [p for p, _, _ in entries])
    y = np.array([lab for _, lab, _ in entries])
    res = {"n_eval": len(entries), "n_bonafide": int((y == 0).sum()),
           "eer": compute_eer(scores, y), "auc": compute_auc(scores, y),
           "per_attack_mean_score": {}}
    for a in sorted({att for _, lab, att in entries if lab == 1}):
        mask = np.array([att == a for _, _, att in entries])
        res["per_attack_mean_score"][a] = float(scores[mask].mean())
    bona = np.array([lab == 0 for _, lab, _ in entries])
    res["per_attack_mean_score"]["bonafide"] = float(scores[bona].mean())
    _merge_results({"clean_eval": res})
    print(f"\nEER {res['eer']:.4f}  AUC {res['auc']:.4f}  "
          f"({res['n_eval']} files, unseen attacks A07-A19)")


def codec_sample():
    """The seeded 600-file (150 bonafide / 450 spoof) eval subsample used by every codec study."""
    entries = _pairs("eval")
    rng = random.Random(SEED)
    bona = [e for e in entries if e[1] == 0]
    spoof = [e for e in entries if e[1] == 1]
    return (rng.sample(bona, min(CODEC_BONA, len(bona)))
            + rng.sample(spoof, min(CODEC_SPOOF, len(spoof))))


def codecs(limit=None):
    sample = codec_sample()
    if limit:
        sample = sample[:limit // 2] + sample[-limit // 2:]  # keep both classes
    y = np.array([lab for _, lab, _ in sample])
    model = _load_model()
    res = {}
    scores = {c: [] for c in CONDITIONS}
    for i, (p, _, _) in enumerate(sample):
        d = degrade_all(p)
        with torch.no_grad():
            for c in CONDITIONS:
                scores[c].append(model(_extract(_load_audio(d[c]))).item())
        cleanup(d)
        if (i + 1) % 100 == 0:
            print(f"degraded+scored {i + 1}/{len(sample)}")
    for c in CONDITIONS:
        s = np.array(scores[c])
        res[c] = {"eer": compute_eer(s, y), "auc": compute_auc(s, y),
                  "mean_score_bonafide": float(s[y == 0].mean()),
                  "mean_score_spoof": float(s[y == 1].mean())}
    _merge_results({"codec_study": {"n": len(sample), "per_condition": res}})
    print(f"\n{'condition':<10} {'EER':>7} {'AUC':>6}")
    for c in CONDITIONS:
        print(f"{c:<10} {res[c]['eer']:>7.4f} {res[c]['auc']:>6.3f}")


def _merge_results(update):
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data.update(update)
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prep":
        prep()
    elif cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "eval":
        evaluate()
    elif cmd == "codecs":
        codecs()
    elif cmd == "all":
        train()
        evaluate()
        codecs()
    elif cmd == "smoke":
        # Note: tiny end-to-end run so the full pipeline never fails an hour in
        prep(limit=40)
        train(epochs=2, limit=40)
        evaluate(limit=40)
        codecs(limit=20)
        print("SMOKE OK")
    else:
        sys.exit(f"unknown command '{cmd}' — use prep|train|eval|codecs|all|smoke")
