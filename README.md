# Local TTS Reader

Local browser reader for long text/webnovel-style chapters with sentence or paragraph chunking, highlighted playback, autoscroll, caching, and pluggable local TTS engines.

Current engines:

- Kokoro on `http://127.0.0.1:7860`
- NeuTTS Nano on `http://127.0.0.1:7861`
- Experimental Qwen3-TTS scripts are included, but Kokoro and NeuTTS are the reader-integrated paths.

## Files

- `kokoro_tts_server.py` serves the reader UI and Kokoro `/api/speak`.
- `kokoro_reader.html` is the browser reader UI.
- `neutts_tts_server.py` serves NeuTTS Nano `/api/speak`.
- `qwen3_tts_server.py` is the earlier Qwen3-TTS local test server.
- `setup.ps1` recreates the Kokoro and NeuTTS virtual environments.
- `start_all.ps1` starts Kokoro and NeuTTS together.
- `run_kokoro_tts_server.ps1`, `run_neutts_tts_server.ps1`, and `run_qwen3_tts_server.ps1` start individual services.
- `requirements-kokoro.txt`, `requirements-neutts.txt`, and `requirements-qwen3.txt` capture Python dependencies.
- `samples/neutts/*.pt` and `samples/neutts/*.txt` are the NeuTTS reference voices used by the reader.

## Local Setup

Create separate virtual environments for Kokoro and NeuTTS. They intentionally stay separate because their phonemizer dependencies conflict.

Recommended setup:

```powershell
.\setup.ps1
```

Manual setup:

Kokoro:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-kokoro.txt
```

NeuTTS:

```powershell
python -m venv .venv-neutts
.\.venv-neutts\Scripts\python.exe -m pip install --upgrade pip
.\.venv-neutts\Scripts\python.exe -m pip install -r requirements-neutts.txt
```

## Run

Start both integrated services:

```powershell
.\start_all.ps1
```

Or start Kokoro:

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

Use the reader's `Engine` select to switch between Kokoro and NeuTTS Nano.

## Reader Features

- Engine status cards for Kokoro and NeuTTS.
- Local session resume through browser storage.
- JSON session export/import for text, settings, current position, and bookmarks.
- Bookmarks tied to the current text.
- NeuTTS custom voice references from uploaded audio, microphone recording, or browser shared-audio capture.
- Auto transcript for custom voice references through a local Whisper ASR endpoint.
- Chunk controls for sentence, paragraph, or line mode with min/max character bounds.
- Queue caching, current-chunk highlighting, autoscroll, and save-current-audio support.

## Notes

- Kokoro is the faster/default path and can use CUDA when the local PyTorch install supports it.
- NeuTTS Nano currently runs through the isolated `.venv-neutts` setup. In this project it is configured for CPU by default because the tested GPU path required a CUDA-enabled `llama-cpp-python` build.
- Creating a custom NeuTTS voice loads the full NeuCodec encoder and can take several minutes the first time. The UI shows elapsed recording/encoding time while it runs.
- Auto transcript uses `openai/whisper-tiny.en` by default. First use may download/load the model and take time; verify the transcript manually before saving the voice.
- Custom NeuTTS references are trimmed to about 12 seconds / 650 speech codes by default so they fit Nano's 2048-token context window. When trimming occurs, the saved reference transcript is trimmed too; best results still come from recording a short, exact 5-12 second reference in the style you want.
- Custom voice reference files are saved under `samples/neutts/custom/` and ignored by Git by default to avoid committing personal voice data accidentally.
- Generated WAV files, logs, model caches, and virtual environments are intentionally ignored by Git.
