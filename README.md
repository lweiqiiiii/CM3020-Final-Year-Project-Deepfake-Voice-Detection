# Deepfake Voice Detector

A local web app that checks whether a caller's voice is human or AI-generated, and a study of whether that check still works after the audio has passed through phone codecs.

Final Year Project, University of London (Goldsmiths), CM3020, Project Template 4.1: Orchestrating AI models to achieve a goal

---

## How to run

### 1. Install the requirements

You need:

- **Python 3.10 – 3.13** (tested on 3.13). Check with `python3 --version`.
- **ffmpeg**, used to read browser recordings and to simulate phone codecs:

| System | Install ffmpeg |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | `winget install ffmpeg`, then open a new terminal |

Check it works with `ffmpeg -version`.

### 2. Download and set up

```bash
git clone <this-repository-url>
cd deepfake-voice-detector

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # about 1 GB, mostly PyTorch
```

On Windows, type `python` instead of `python3`.

### 3. Start the app

```bash
python app.py
```

Wait for `Running on http://127.0.0.1:5001`, then open **http://localhost:5001** in your browser.

The trained model weights are included in the repository (`models/model_pilot_vad.pth` and `aasist/AASIST.pth`), so nothing else needs downloading or training. Stop the app with **Ctrl+C**.

### 4. Use it

- **Record the caller**: put a call on speaker, press the button, and let them talk for 5–10 seconds. Your browser will ask for microphone permission.
- **Or upload a recording**: WAV, MP3, M4A, FLAC, WebM or OGG, up to 25 MB.

The result is **Likely AI voice**, **Sounds human** or **Unsure**, with advice on what to do. Open *How we decided* to see each model's score, and *For researchers* to re-test the clip through four phone codecs (the "codec gauntlet").

### Check the installation (optional)

Each module has a self-check:

```bash
python detect.py      # Pipeline OK
python degrade.py     # ffmpeg found ... / Available codecs
python evaluate.py    # Evaluate OK
python vad.py         # VAD OK
python ensemble.py    # Ensemble OK
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `Port 5001 is in use` | Another program is using the port. On macOS, turn off AirPlay Receiver or stop the other app. |
| "Could not analyse this file" on a browser recording | ffmpeg is missing or not on your PATH. Check `ffmpeg -version`. |
| Microphone button does nothing | Recording only works on `http://localhost` (or HTTPS). Open the page on the same computer that runs `app.py`. |
| `ModuleNotFoundError: torch` | The virtual environment isn't active. Run `source .venv/bin/activate` again. |

---

## How it works

```
recording ─► ffmpeg ─► Silero VAD ──speech only──► LCNN (trained here) ──► verdict
                           │                                                  ▲
                           └──full clip──► AASIST (pre-trained) ── second opinion
```

| Model | Role |
|---|---|
| ElevenLabs TTS (pre-trained) | Generated the AI-voice clones used for training |
| Silero VAD (pre-trained) | Removes silence so the detector judges the voice, not the pauses |
| AASIST (pre-trained) | State-of-the-art detector, shown as a second opinion |
| LCNN on LFCC features (trained in this project) | Primary detector |

## Key results

Measured on the ASVspoof 2019 LA benchmark. EER is the equal error rate: lower is better, and 50% is chance.

| Detector | Unseen attacks | Worst phone codec | Silence removed |
|---|---|---|---|
| LCNN, clean-trained | 17.5% | 22.3% (G.711) | 50.5% |
| LCNN, codec-augmented | 16.6% | 19.0% | 50.2% |
| AASIST (pre-trained) | 0.9% | 2.0% (G.726) | 23.6% |

- 8 kHz phone codecs hurt detection most, and training on codec-degraded audio fixes it.
- Much of the benchmark accuracy came from silence, not voice, so the app scores speech only.
- Full numbers are in [`results/results_final.json`](results/results_final.json), and the discussion is in the report.

## Reproducing the experiments (optional)

This is not needed to run the app. It needs the [ASVspoof 2019 LA dataset](https://datashare.ed.ac.uk/handle/10283/3336) (7.6 GB), placed at `../datasets/asvspoof2019/LA.zip`, and several hours of CPU time:

```bash
python asvspoof.py prep           # sample the benchmark subset
python final_study.py degrade     # codec versions of every file
python final_study.py vadcache    # speech-only versions
python final_study.py lcnn        # train LCNNs (3 seeds)
python final_study.py lcnn_vad
python final_study.py aasist      # score with pre-trained AASIST
python final_study.py report      # → results/results_final.json
python final_study.py bootstrap   # 95% confidence intervals → results/results_bootstrap.json
python final_figures.py all       # → results/*.png
```



