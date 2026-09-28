import json
import os
import subprocess
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import soundfile as sf
import torch

from detect import LCNN, CLIP_SAMPLES, SAMPLE_RATE, _read, score_windows, tile
from degrade import degrade_all, cleanup
from ensemble import aasist_input, aasist_prob
from vad import speech_only

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024   # ~10 min of WAV; a call clip is far smaller
HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "tmp", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Models load once per process. AASIST and Silero cache themselves on first use
# (ensemble._model / vad._model); the LCNN is loaded here.
LCNN_WEIGHTS = os.path.join(HERE, "models", "model_pilot_vad.pth")   # pilot LCNN, codec-augmented, trained on VAD-trimmed + tiled clips
lcnn = None
if os.path.exists(LCNN_WEIGHTS):
    lcnn = LCNN()
    lcnn.load_state_dict(torch.load(LCNN_WEIGHTS, map_location="cpu", weights_only=True))
    lcnn.eval()


DECISION_RULE = "LCNN (pilot-trained on speech-only audio); AASIST shown as a second opinion"   # shown in the UI
MIN_SPEECH_S = 1.0   # under this much speech there is nothing to judge — matches BANDS.MIN_SPEECH_S in index.html


def decision_score(p_lcnn: float, p_aasist: float) -> float:
    """The one verdict rule: the score thresholded at 0.5 (fusion would be `return fuse(p_lcnn, p_aasist)`).

    LCNN-primary, from measured numbers on the target channel (phone/laptop-mic
    speech, pilot corpus): pre-trained AASIST EER 26.3% and flags 6/19 real clips
    (cross_corpus.json); mean fusion let those false alarms drag real voices to ~0.5.
    The LCNN is the pilot model retrained on VAD-trimmed + tiled clips with codec
    augmentation: the untrimmed pilot corpus is separable on leading silence alone
    (real ~2 s vs fake ~0.03 s, EER 0.000), and with silence removed the codec-aug
    variant still gets seeded 5-fold CV EER 0.000 under every codec
    (results_pilot_vad.json), whereas the clean-trained one misses 9/19 fakes under G.711.
    Note: single speaker, one TTS generator — retrain on more speakers/channels before trusting it widely.
    """
    return p_lcnn


def analyse(path: str) -> dict:
    """Silero VAD -> LCNN on speech-only audio + AASIST on the untrimmed clip -> decision_score verdict."""
    if lcnn is None:
        raise FileNotFoundError(LCNN_WEIGHTS)
    wav = _read(path)                            # 16 kHz mono, untrimmed
    speech = speech_only(wav)
    # speech_only returns its input object unchanged when it finds no speech
    speech_seconds = 0.0 if speech is wav else speech.shape[1] / SAMPLE_RATE
    # LCNN sees what it was trained on: speech only, tiled to 4 s (no silence or
    # zero-pad length to key on), first 4 s. Multi-window averaging was measured
    # and rejected (results_multiwindow.json), so n_windows is 1.
    p_lcnn, n_windows = score_windows(lcnn, tile(speech)[:, :CLIP_SAMPLES])
    # AASIST keeps the untrimmed clip: it was trained on untrimmed ASVspoof audio and
    # trimming raised its false alarms on real speech (pilot EER 26.3% -> 31.6%,
    # mean spoof prob on real 0.51 -> 0.90, cross_corpus.json) — Müller et al. 2021.
    # Note: aasist_input re-reads the file (cheap) rather than duplicating its tiling
    p_aasist = aasist_prob(aasist_input(path)).item()
    score = decision_score(p_lcnn, p_aasist)
    return {
        "fake_probability": round(score, 4),
        # silence alone scores ~1.0 on the LCNN, so no/too little speech must not read as FAKE
        "verdict": "UNSURE" if speech_seconds < MIN_SPEECH_S else ("FAKE" if score > 0.5 else "REAL"),
        "decision_rule": DECISION_RULE,
        "lcnn_model": os.path.basename(LCNN_WEIGHTS),
        "lcnn_probability": round(p_lcnn, 4),
        "aasist_probability": round(p_aasist, 4),
        "speech_seconds": round(speech_seconds, 2),
        "n_windows": n_windows,
        "detectors_agree": (p_lcnn > 0.5) == (p_aasist > 0.5),
    }


MISSING_MODEL = "The detector model is missing (models/model_pilot_vad.pth). Restore it from the repository or retrain with: python3 run_cv_study.py --vad"
BAD_AUDIO = "Could not analyse this file. Please upload a valid audio recording (WAV, MP3, M4A or FLAC)."


def save_upload(file) -> str:
    # unique name per request: every browser recording is called call-recording.webm,
    # so a fixed name would let concurrent requests overwrite or delete each other's audio
    ext = os.path.splitext(secure_filename(file.filename or ""))[1]
    fd, path = tempfile.mkstemp(dir=UPLOAD_DIR, suffix=ext)
    os.close(fd)
    file.save(path)
    # Browser recordings arrive as webm/mp4 and soundfile can't read those —
    # transcode anything unreadable to 16 kHz mono WAV via ffmpeg.
    try:
        sf.info(path)
    except Exception:
        wav_path = path + ".wav"
        try:
            subprocess.run(
                # untrusted input: local files only (no playlist/concat fetching of URLs), bounded time
                ["ffmpeg", "-y", "-protocol_whitelist", "file", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
                check=True, capture_output=True, timeout=60,
            )
        except Exception:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return path  # let analyse() fail → routes return the friendly 400
        os.remove(path)
        path = wav_path
    return path


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "That file is too large (limit 25 MB). Record or upload 5–10 seconds of the call."}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results")
def results():
    try:
        with open(os.path.join(HERE, "results", "results_cv.json")) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "results_cv.json not found — run: python3 run_cv_study.py"}), 404


@app.route("/results/final")
def results_final():
    try:
        with open(os.path.join(HERE, "results", "results_final.json")) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "results_final.json not found — run: python3 final_study.py report"}), 404


@app.route("/detect", methods=["POST"])
def detect():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    if lcnn is None:
        return jsonify({"error": MISSING_MODEL}), 503
    path = save_upload(request.files["audio"])
    try:
        result = analyse(path)
    except Exception:
        return jsonify({"error": BAD_AUDIO}), 400
    finally:
        # Note: cleanup is synchronous — fine for a demo with one user;
        # swap for a background task queue (e.g. Celery) in production.
        os.remove(path)

    return jsonify(result)


@app.route("/demo", methods=["POST"])
def demo():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    if lcnn is None:
        return jsonify({"error": MISSING_MODEL}), 503
    path = save_upload(request.files["audio"])
    degraded_paths = {}
    try:
        clean_result = analyse(path)
        degraded_paths = degrade_all(path)

        results = {"clean": clean_result}
        for codec, dpath in degraded_paths.items():
            if codec != "clean":
                results[codec] = analyse(dpath)
    except Exception:
        return jsonify({"error": BAD_AUDIO}), 400
    finally:
        os.remove(path)
        if degraded_paths:
            cleanup(degraded_paths)

    return jsonify(results)


if __name__ == "__main__":
    # debug off by default: the Werkzeug debugger executes code. Opt in with FLASK_DEBUG=1.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
