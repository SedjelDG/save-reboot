# Local Qwen3-TTS Smoke Test

This folder contains a local install of the Qwen3-TTS Hugging Face Space code plus a minimal runner.

## What is installed

- `.venv/`: local Python virtual environment
- `qwen3-tts/`: cloned Qwen3-TTS Space code
- `.hf-cache/`: local Hugging Face model cache
- `smoke_qwen3_tts.py`: minimal one-model test script
- `run_qwen3_tts.ps1`: PowerShell wrapper

## Run

```powershell
.\run_qwen3_tts.ps1 --text "Hello. This is a short test." --output qwen3_tts_80tokens.wav --max-new-tokens 80 --no-subtalker-sample
```

## Run The Browser Server

```powershell
.\run_qwen3_tts_server.ps1 --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

The browser page calls:

```text
POST http://127.0.0.1:7860/api/speak
```

The server loads the model lazily on the first request and keeps it warm afterward.

To stop the server:

```powershell
.\stop_qwen3_tts_server.ps1
```

## Run Kokoro Instead

Kokoro is much faster on this GTX 1650 than Qwen3-TTS and is the better fit for the reader prototype.

Smoke test:

```powershell
.\.venv\Scripts\python.exe .\smoke_kokoro_tts.py --text "Hello. This is Kokoro running locally." --voice af_heart --output kokoro_test.wav
```

Browser server:

```powershell
.\run_kokoro_tts_server.ps1 --host 127.0.0.1 --port 7860 --preload
```

Then open:

```text
http://127.0.0.1:7860
```

Current tested Kokoro result: about 3.4 seconds of audio generated through the localhost API in about 3.6 seconds.

The current reader UI lives in:

```text
kokoro_reader.html
```

It supports source/read tabs, text file import, sentence/paragraph chunking, queue playback, current-chunk highlighting, autoscroll, prefetch, audio caching, voice/language/speed/volume controls, previous/next/stop/pause, current chunk download, and a test tone.

## NeuTTS Air Test

NeuTTS Air Q4 works locally through the ONNX decoder path with pre-encoded reference audio:

```powershell
$env:TEMP='C:\Users\Administrator\Desktop\the reader\.tmp'
$env:TMP='C:\Users\Administrator\Desktop\the reader\.tmp'
$env:PHONEMIZER_ESPEAK_LIBRARY='C:\Users\Administrator\Desktop\the reader\.venv\Lib\site-packages\neutts\espeak-ng.dll'
.\.venv\Scripts\python.exe .\smoke_neutts_air.py --text "This is NeuTTS Air voice cloning running locally." --output neutts_air_output.wav
```

Results on this PC:

- Q4 + ONNX decoder generated valid audio with the bundled `jo` reference.
- After model load, repeated chunks took about 8-12 seconds for about 4-5 seconds of audio.
- NeuTTS and Kokoro fight over the `phonemizer` package in one venv, so NeuTTS should be isolated in its own venv before wiring it into the reader.

The default model is:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

## Notes From This PC

- CUDA works with the local venv and sees the GTX 1650.
- The model loads on the GPU.
- CUDA `float16` generated broken constant audio on this GTX 1650.
- CUDA `float32` generated a real waveform and is the server default now.
- Full sampling crashed on this GPU with a CUDA device-side assertion.
- Running with `--no-subtalker-sample` succeeds.
- `max-new-tokens` behaves like an audio length cap. On the 12 Hz model, `80` tokens produced about 6.3 seconds, while `512` produced about 40.8 seconds.
- The Python `sox` package is installed, but the separate SoX executable is not on PATH. The smoke test still generated WAV files.
- The localhost server was tested at `http://127.0.0.1:7860` and generated `server_test.wav` via `/api/speak`.

## Useful Speakers

```text
Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
```

## Useful Languages

```text
Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian
```
