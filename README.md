# Local TTS Reader

Local browser reader for long text/webnovel-style chapters with sentence or paragraph chunking, highlighted playback, autoscroll, caching, and pluggable local TTS engines.

Current engines:

- Kokoro on `http://127.0.0.1:7860`
- NeuTTS Air on `http://127.0.0.1:7861`
- Experimental Qwen3-TTS scripts are included, but Kokoro and NeuTTS are the reader-integrated paths.

## Files

- `kokoro_tts_server.py` serves the reader UI and Kokoro `/api/speak`.
- `kokoro_reader.html` is the browser reader UI.
- `neutts_tts_server.py` serves NeuTTS Air `/api/speak`.
- `qwen3_tts_server.py` is the earlier Qwen3-TTS local test server.
- `run_kokoro_tts_server.ps1`, `run_neutts_tts_server.ps1`, and `run_qwen3_tts_server.ps1` start the local services.
- `samples/neutts/*.pt` and `samples/neutts/*.txt` are the NeuTTS reference voices used by the reader.

## Local Setup

Create separate virtual environments for Kokoro and NeuTTS. They intentionally stay separate because their phonemizer dependencies conflict.

Kokoro:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install kokoro soundfile numpy torch
```

NeuTTS:

```powershell
python -m venv .venv-neutts
.\.venv-neutts\Scripts\python.exe -m pip install --upgrade pip
.\.venv-neutts\Scripts\python.exe -m pip install "neutts[all]"
```

## Run

Start Kokoro:

```powershell
.\run_kokoro_tts_server.ps1
```

Start NeuTTS in a second terminal:

```powershell
.\run_neutts_tts_server.ps1
```

Open:

```text
http://127.0.0.1:7860
```

Use the reader's `Engine` select to switch between Kokoro and NeuTTS Air.

## Notes

- Kokoro is the faster/default path and can use CUDA when the local PyTorch install supports it.
- NeuTTS Air currently runs through the isolated `.venv-neutts` setup. In this project it is configured for CPU by default because the tested GPU path required a CUDA-enabled `llama-cpp-python` build.
- Generated WAV files, logs, model caches, and virtual environments are intentionally ignored by Git.
