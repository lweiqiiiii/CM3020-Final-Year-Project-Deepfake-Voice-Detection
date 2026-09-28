"""Final-report experiments on the ASVspoof 2019 LA subset (needs `asvspoof.py prep` done).

  python3 final_study.py degrade   # cache 4 codec versions of train (2580) + codec sample (600) (~15 min)
  python3 final_study.py vadcache  # speech-only (Silero, after codec) copies of every file above + eval
  python3 final_study.py lcnn      # LCNN clean vs codec-augmented x 3 seeds, all conditions  (~50 min)
  python3 final_study.py aasist    # pre-trained AASIST, all conditions                        (~50 min)
  python3 final_study.py lcnn_vad  # LCNN codec-augmented on the VAD cache x 3 seeds (needs vadcache)
  python3 final_study.py report    # fusion, per-attack EER, seed spread -> results_final.json
  python3 final_study.py vadfair   # re-score the seed-0 clean/aug LCNNs on tiled (not zero-padded) speech -> results_vadfair.json
  python3 final_study.py bootstrap # 95% CIs for every EER + paired differences -> results_bootstrap.json (~3 min)
  python3 final_study.py smoke     # tiny end-to-end run of lcnn + aasist + report

lcnn and aasist are independent — run them in parallel after degrade.
Conditions scored for every model:
  eval_clean   full 2394-file unseen-attack eval set (per-attack EER from it)
  <codec>      600-file codec sample after clean/g711/g726/mp3_low/opus_low round-trip
  vad          600-file sample, clean, Silero-VAD speech-only (tests the ASVspoof silence cue)
  (lcnn_vad reads every condition from the VAD cache, so for it vad == clean)
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from asvspoof import SUBSET, CONDITIONS, _pairs, codec_sample
from degrade import apply_codec
from detect import _train_model, _extract, _load_audio, tile, _read
from evaluate import compute_eer, compute_auc

HERE = os.path.dirname(os.path.abspath(__file__))
CODEC_DIR = os.path.join(SUBSET, "codec")
VAD_DIR = os.path.join(SUBSET, "vad")
SCORES = os.path.join(HERE, "results", "scores_final")      # one json per model, so lcnn/aasist can run in parallel
RESULTS = os.path.join(HERE, "results", "results_final.json")
SEEDS = [0, 1, 2]
EPOCHS = 10
CODECS = CONDITIONS[1:]


def _cpath(path, codec):
    return os.path.join(CODEC_DIR, codec, os.path.basename(path).replace(".flac", ".wav"))


def _vpath(path, cond):
    return os.path.join(VAD_DIR, cond, os.path.basename(path).replace(".flac", ".wav"))


def _variant(path, cond):
    return path if cond in ("clean", "vad", "eval_clean") else _cpath(path, cond)


def _vad_variant(path, cond):
    """Cached speech-only file; eval_clean, clean and vad all read vad/clean (vad == clean here)."""
    return _vpath(path, cond if cond in CODECS else "clean")


def degrade():
    todo = [(p, c) for p, _, _ in _pairs("train") + codec_sample() for c in CODECS
            if not os.path.exists(_cpath(p, c))]
    for c in CODECS:
        os.makedirs(os.path.join(CODEC_DIR, c), exist_ok=True)
    print(f"{len(todo)} codec round-trips to run")
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(lambda pc: apply_codec(pc[0], pc[1], _cpath(*pc)), todo)):
            if (i + 1) % 1000 == 0:
                print(f"degraded {i + 1}/{len(todo)}", flush=True)


def vadcache():
    import soundfile as sf
    from vad import speech_only
    torch.set_num_threads(2)       # the lcnn/aasist jobs own the other cores
    todo = [(p, "clean") for p, _, _ in _pairs("train") + _pairs("eval")]
    todo += [(p, c) for p, _, _ in _pairs("train") + codec_sample() for c in CODECS]
    names = {os.path.basename(p) for p, _, _ in _pairs("train") + _pairs("eval")}
    assert len(names) == len(_pairs("train")) + len(_pairs("eval")), "basenames collide in vad/clean"
    for c in CONDITIONS:
        os.makedirs(os.path.join(VAD_DIR, c), exist_ok=True)
    todo = [(p, c) for p, c in todo if not os.path.exists(_vpath(p, c))]
    print(f"{len(todo)} files to VAD-trim", flush=True)
    label = {p: y for p, y, _ in _pairs("train") + _pairs("eval")}
    t0 = time.time()
    with open(os.path.join(VAD_DIR, "trim_stats.jsonl"), "a") as log:   # appends across resumed runs
        for i, (p, c) in enumerate(todo):
            wav = _read(p if c == "clean" else _cpath(p, c))    # VAD after the codec, as on a real call
            sp = speech_only(wav)
            dst = _vpath(p, c)
            sf.write(dst + ".part", tile(sp)[0].numpy(), 16000, format="WAV")
            os.replace(dst + ".part", dst)                      # a killed run never leaves a half file
            log.write(json.dumps([c, label[p], wav.shape[1], sp.shape[1], sp is wav]) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"vad {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    st = np.array([json.loads(l)[1:] for l in open(os.path.join(VAD_DIR, "trim_stats.jsonl"))], dtype=float)
    print(f"done in {time.time() - t0:.0f}s; cache has {len(st)} logged files; "
          f"no speech found in {st[:, 3].mean():.4f}")
    for y, name in ((0, "bonafide"), (1, "spoof")):
        k = st[:, 0] == y
        print(f"{name}: mean fraction trimmed {1 - (st[k, 2] / st[k, 1]).mean():.3f} (n={int(k.sum())})")


def _eval_sets(limit=None):
    ev, cs = _pairs("eval"), codec_sample()
    if limit:
        ev, cs = ev[:limit // 2] + ev[-limit // 2:], cs[:limit // 2] + cs[-limit // 2:]
    sets = {"eval_clean": ev, "vad": cs}
    sets.update({c: cs for c in CONDITIONS})
    return sets


def _save_scores(model_name, scores):
    os.makedirs(SCORES, exist_ok=True)
    json.dump(scores, open(os.path.join(SCORES, f"{model_name}.json"), "w"))


def lcnn(limit=None, kinds=("clean", "aug")):
    train = [(p, y) for p, y, _ in _pairs("train")]
    if limit:
        train = train[:limit // 2] + train[-limit // 2:]
    files_for = {"clean": train,
                 "aug": train + [(_cpath(p, c), y) for p, y in train for c in CODECS],
                 # VAD-trimmed + codec-augmented: speech-only clean + 4 speech-only codec versions
                 "vad": [(_vpath(p, c), y) for p, y in train for c in CONDITIONS]}
    sets = _eval_sets(limit)
    for kind in kinds:
        files = files_for[kind]
        for seed in SEEDS:
            torch.manual_seed(seed)
            print(f"\n== LCNN {kind} seed {seed}: {len(files)} training items", flush=True)
            model, _, acc = _train_model(list(files), epochs=2 if limit else EPOCHS,
                                         rng=random.Random(seed))
            model.eval()
            if seed == 0 and not limit:          # smoke runs (limit set) must not replace real weights
                torch.save(model.state_dict(), os.path.join(HERE, "models", f"model_asvspoof_{kind}.pth"))
            out = {}
            with torch.no_grad():
                for cond, entries in sets.items():
                    if kind == "vad":   # already trimmed + tiled in the cache
                        out[cond] = [model(_extract(_load_audio(_vad_variant(p, cond)))).item()
                                     for p, _, _ in entries]
                    else:
                        out[cond] = [model(_extract(_load_audio(_variant(p, cond), vad=cond == "vad"))).item()
                                     for p, _, _ in entries]
            _save_scores(f"lcnn_{kind}_s{seed}", out)
            print(f"   train_acc={acc:.3f}  eval_clean EER="
                  f"{compute_eer(np.array(out['eval_clean']), np.array([y for _, y, _ in sets['eval_clean']])):.4f}",
                  flush=True)


def lcnn_vad(limit=None):
    lcnn(limit, kinds=("vad",))


def aasist(limit=None):
    from ensemble import aasist_input, aasist_prob
    torch.set_num_threads(4)       # leave cores for a parallel `lcnn` run
    out = {}
    for cond, entries in _eval_sets(limit).items():
        s = []
        for i in range(0, len(entries), 16):
            batch = torch.cat([aasist_input(_variant(p, cond), vad=cond == "vad")
                               for p, _, _ in entries[i:i + 16]])
            s += aasist_prob(batch).tolist()
        out[cond] = s
        print(f"AASIST {cond}: {len(s)} scored", flush=True)
    _save_scores("aasist", out)


def _metrics(scores, entries):
    s, y = np.array(scores), np.array([lab for _, lab, _ in entries])
    return {"eer": compute_eer(s, y), "auc": compute_auc(s, y),
            "mean_bonafide": float(s[y == 0].mean()), "mean_spoof": float(s[y == 1].mean())}


def _per_attack(scores, entries):
    s = np.array(scores)
    att = np.array([a for _, _, a in entries])
    bona = att == "-"
    out = {}
    for a in sorted(set(att) - {"-"}):
        m = bona | (att == a)
        out[a] = compute_eer(s[m], (att[m] != "-").astype(int))
    return out


def report(limit=None):
    sc = {f[:-5]: json.load(open(os.path.join(SCORES, f))) for f in sorted(os.listdir(SCORES))}
    sets = _eval_sets(limit)
    fused = {}
    for kind in ("clean", "aug", "vad"):
        if f"lcnn_{kind}_s0" not in sc:
            continue
        fused[f"fusion_{kind}"] = {c: [(a + b) / 2 for a, b in zip(sc["aasist"][c], sc[f"lcnn_{kind}_s0"][c])]
                                   for c in sets}
    sc.update(fused)
    res = {"n": {k: len(v) for k, v in sets.items() if k in ("eval_clean", "clean")}, "models": {}}
    for name, per in sc.items():
        res["models"][name] = {c: _metrics(per[c], sets[c]) for c in sets}
        res["models"][name]["per_attack_eer"] = _per_attack(per["eval_clean"], sets["eval_clean"])
    # seed spread: mean ± std of EER over seeds, per LCNN kind and condition
    res["seed_summary"] = {}
    for kind in ("clean", "aug", "vad"):
        runs = [res["models"][f"lcnn_{kind}_s{s}"] for s in SEEDS if f"lcnn_{kind}_s{s}" in res["models"]]
        if not runs:
            continue
        res["seed_summary"][f"lcnn_{kind}"] = {
            c: {"mean": float(np.mean([r[c]["eer"] for r in runs])),
                "std": float(np.std([r[c]["eer"] for r in runs])), "n_seeds": len(runs)}
            for c in sets}
    json.dump(res, open(RESULTS, "w"), indent=1)
    conds = ["eval_clean"] + CONDITIONS + ["vad"]
    print(f"{'model':<16}" + "".join(f"{c:>11}" for c in conds))
    for name, m in res["models"].items():
        print(f"{name:<16}" + "".join(f"{m[c]['eer']:>11.4f}" for c in conds))
    print(f"wrote {RESULTS}")


def vadfair():
    """The 'vad' column zero-pads trimmed speech for the untrimmed-trained LCNNs but tiles it for
    AASIST (and lcnn_vad). Re-score the seed-0 clean/aug LCNNs on the tiled speech-only cache so
    every detector sees the same input. Only seed-0 weights were saved, so this is seed 0 only."""
    from detect import LCNN
    entries = codec_sample()
    y = np.array([lab for _, lab, _ in entries])
    sc = {f[:-5]: json.load(open(os.path.join(SCORES, f))) for f in sorted(os.listdir(SCORES))}
    out = {"n": len(entries), "note": vadfair.__doc__.split("\n")[0], "models": {}}
    for kind in ("clean", "aug"):
        m = LCNN()
        m.load_state_dict(torch.load(os.path.join(HERE, "models", f"model_asvspoof_{kind}.pth"),
                                     map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            tiled = np.array([m(_extract(_load_audio(_vad_variant(p, "vad")))).item() for p, _, _ in entries])
        padded = np.array(sc[f"lcnn_{kind}_s0"]["vad"])
        out["models"][f"lcnn_{kind}_s0"] = {
            "zero_padded": {"eer": compute_eer(padded, y), "auc": compute_auc(padded, y),
                            "share_scores_equal_1": float((padded == 1.0).mean())},
            "tiled": {"eer": compute_eer(tiled, y), "auc": compute_auc(tiled, y),
                      "share_scores_equal_1": float((tiled == 1.0).mean()),
                      "mean_bonafide": float(tiled[y == 0].mean()), "mean_spoof": float(tiled[y == 1].mean())},
            "clean_untrimmed_eer": compute_eer(np.array(sc[f"lcnn_{kind}_s0"]["clean"]), y)}
        print(kind, json.dumps(out["models"][f"lcnn_{kind}_s0"]))
    json.dump(out, open(os.path.join(HERE, "results", "results_vadfair.json"), "w"), indent=1)
    print("wrote results/results_vadfair.json")


def bootstrap(reps=1000, seed=0):
    """Stratified bootstrap over files (bona fide and spoof resampled separately, 2.5/97.5
    percentiles). The 600-file codec conditions share one resample per replicate, so codec-vs-clean
    and augmented-vs-clean differences are paired. LCNN rows: EER of the 3-seed mean, i.e. the
    CI covers test-set sampling, not seed variance (that is seed_summary's std)."""
    sc = {f[:-5]: json.load(open(os.path.join(SCORES, f))) for f in sorted(os.listdir(SCORES))}
    sets = _eval_sets()
    sc["fusion_clean"] = {c: [(a + b) / 2 for a, b in zip(sc["aasist"][c], sc["lcnn_clean_s0"][c])] for c in sets}
    groups = {"lcnn_clean": [f"lcnn_clean_s{s}" for s in SEEDS], "lcnn_aug": [f"lcnn_aug_s{s}" for s in SEEDS],
              "lcnn_vad": [f"lcnn_vad_s{s}" for s in SEEDS], "aasist": ["aasist"], "fusion_clean": ["fusion_clean"]}
    conds = ["eval_clean"] + CONDITIONS + ["vad"]
    S = {g: {c: np.array([sc[m][c] for m in ms]) for c in conds} for g, ms in groups.items()}   # (n_seeds, n_files)
    y = {c: np.array([lab for _, lab, _ in sets[c]]) for c in conds}
    idx0 = {k: (np.flatnonzero(y[c] == 0), np.flatnonzero(y[c] == 1)) for k, c in (("eval", "eval_clean"), ("sample", "clean"))}
    eer = lambda g, c, ix: float(np.mean([compute_eer(row[ix], y[c][ix]) for row in S[g][c]]))
    point = {g: {c: eer(g, c, np.arange(len(y[c]))) for c in conds} for g in groups}
    rng = np.random.default_rng(seed)
    draws = {g: {c: [] for c in conds} for g in groups}
    for r in range(reps):
        ix = {k: np.concatenate([rng.choice(b, len(b)), rng.choice(s, len(s))]) for k, (b, s) in idx0.items()}
        for g in groups:
            for c in conds:
                draws[g][c].append(eer(g, c, ix["eval" if c == "eval_clean" else "sample"]))
        if (r + 1) % 200 == 0:
            print(f"bootstrap {r + 1}/{reps}", flush=True)
    D = {g: {c: np.array(v) for c, v in d.items()} for g, d in draws.items()}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    res = {"reps": reps, "method": bootstrap.__doc__.strip(), "eer": {}, "paired_differences": {}}
    for g in groups:
        res["eer"][g] = {c: {"point": point[g][c], "ci95": ci(D[g][c])} for c in conds}
        for c in CODECS + ["vad"]:          # degradation vs clean on the same resampled files
            d = D[g][c] - D[g]["clean"]
            res["paired_differences"][f"{g}: {c} - clean"] = {
                "point": point[g][c] - point[g]["clean"], "ci95": ci(d), "excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0)}
    for c in conds:                         # does codec augmentation help? same files, both models
        d = D["lcnn_aug"][c] - D["lcnn_clean"][c]
        res["paired_differences"][f"lcnn_aug - lcnn_clean: {c}"] = {
            "point": point["lcnn_aug"][c] - point["lcnn_clean"][c], "ci95": ci(d), "excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0)}
    json.dump(res, open(os.path.join(HERE, "results", "results_bootstrap.json"), "w"), indent=1)
    for k, v in res["paired_differences"].items():
        print(f"{k:<36} {v['point'] * 100:+6.2f}  [{v['ci95'][0] * 100:+6.2f}, {v['ci95'][1] * 100:+6.2f}]  {'*' if v['excludes_zero'] else ''}")
    print("wrote results/results_bootstrap.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "smoke":
        # scratch outputs: smoke must never overwrite the committed results/ or models/
        import tempfile
        SEEDS = [0]
        SCORES = tempfile.mkdtemp(prefix="smoke_scores_")
        RESULTS = os.path.join(SCORES, "results_final.json")
        degrade_needed = not os.path.isdir(CODEC_DIR)
        if degrade_needed:
            sys.exit("run `degrade` first")
        lcnn(limit=40); aasist(limit=40); report(limit=40)
        print("SMOKE OK")
    elif cmd in ("degrade", "vadcache", "lcnn", "lcnn_vad", "aasist", "report", "vadfair", "bootstrap"):
        globals()[cmd]()
    else:
        sys.exit(__doc__)
