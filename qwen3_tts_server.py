import argparse
import io
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
QWEN_SPACE = ROOT / "qwen3-tts"
HF_HOME = ROOT / ".hf-cache"

sys.path.insert(0, str(QWEN_SPACE))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from qwen_tts import Qwen3TTSModel  # noqa: E402


SPEAKERS = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_fu",
    "Vivian",
]

LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
]


def normalize_audio(wav, eps=1e-12, clip=True) -> np.ndarray:
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        peak = np.max(np.abs(y)) if y.size else 0.0
        if peak > 1.0 + 1e-6:
            y = y / (peak + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def make_test_tone(sample_rate=24000, seconds=1.0, frequency=440.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    wav = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    audio = io.BytesIO()
    sf.write(audio, wav, sample_rate, format="WAV", subtype="PCM_16")
    return audio.getvalue()

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen3-TTS Local Reader Test</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f5f2;
      color: #1f2523;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(245, 245, 242, 0.96)),
        #f5f5f2;
    }

    main {
      width: min(980px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.15;
      font-weight: 760;
    }

    .status {
      min-width: 190px;
      padding: 9px 12px;
      border: 1px solid #cbd5cf;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.72);
      font-size: 13px;
      text-align: center;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }

    textarea {
      width: 100%;
      min-height: 360px;
      resize: vertical;
      padding: 18px;
      border: 1px solid #b9c5bd;
      border-radius: 8px;
      background: #fffef9;
      color: #202522;
      font: 18px/1.7 Georgia, "Times New Roman", serif;
    }

    aside {
      display: grid;
      gap: 12px;
      padding: 14px;
      border: 1px solid #cbd5cf;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.76);
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 650;
    }

    select,
    input {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid #b9c5bd;
      border-radius: 7px;
      background: #fffef9;
      color: #202522;
      font: inherit;
    }

    .check {
      grid-template-columns: 18px 1fr;
      align-items: center;
      font-weight: 560;
    }

    .check input {
      width: 18px;
      height: 18px;
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      background: #245e4f;
      color: white;
      font-weight: 740;
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.64;
    }

    .secondary {
      background: #626c66;
    }

    audio {
      width: 100%;
      margin-top: 16px;
    }

    pre {
      min-height: 92px;
      margin: 0;
      padding: 12px;
      overflow: auto;
      border: 1px solid #d3d9d5;
      border-radius: 8px;
      background: #1f2523;
      color: #dfe8e2;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    @media (max-width: 760px) {
      header,
      .layout {
        display: grid;
        grid-template-columns: 1fr;
      }

      .status {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Qwen3-TTS Local Reader Test</h1>
      <div class="status" id="status">Checking server...</div>
    </header>

    <div class="layout">
      <section>
        <textarea id="text">Hello. This is a short test.</textarea>
        <audio id="audio" controls></audio>
      </section>

      <aside>
        <label>
          Speaker
          <select id="speaker"></select>
        </label>
        <label>
          Language
          <select id="language"></select>
        </label>
        <label>
          Max audio tokens
          <input id="maxTokens" type="number" min="12" max="2048" step="4" value="80" />
        </label>
        <label class="check">
          <input id="talkerSample" type="checkbox" checked />
          Main sampler
        </label>
        <label class="check">
          <input id="subtalkerSample" type="checkbox" />
          Subtalker sampler
        </label>
        <button id="speak">Generate & Play</button>
        <button id="tone" type="button" class="secondary">Test Tone</button>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>

  <script>
    const speakers = __SPEAKERS__;
    const languages = __LANGUAGES__;
    const $ = (id) => document.getElementById(id);

    function fillSelect(id, values, selected) {
      const select = $(id);
      select.innerHTML = values.map((value) => {
        const attr = value === selected ? " selected" : "";
        return `<option value="${value}"${attr}>${value}</option>`;
      }).join("");
    }

    function log(message) {
      $("log").textContent = message;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        $("status").textContent = data.model_loaded ? "Model loaded" : "Ready, model loads on first request";
      } catch (error) {
        $("status").textContent = "Server unavailable";
      }
    }

    async function speak() {
      const button = $("speak");
      button.disabled = true;
      $("status").textContent = "Generating...";
      log("Sending request. First generation may take a while.");

      const started = performance.now();
      try {
        const response = await fetch("/api/speak", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: $("text").value,
            speaker: $("speaker").value,
            language: $("language").value,
            max_new_tokens: Number($("maxTokens").value),
            do_sample: $("talkerSample").checked,
            subtalker_dosample: $("subtalkerSample").checked
          })
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText);
        }

        const elapsed = ((performance.now() - started) / 1000).toFixed(1);
        const metadata = response.headers.get("X-Qwen3-TTS-Metadata");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        $("audio").src = url;
        await $("audio").play();
        $("status").textContent = "Playing";
        log(`Generated ${Math.round(blob.size / 1024)} KB in ${elapsed}s.\n${metadata || ""}`);
      } catch (error) {
        $("status").textContent = "Error";
        log(String(error.message || error));
      } finally {
        button.disabled = false;
        refreshHealth();
      }
    }

    async function testTone() {
      $("status").textContent = "Playing tone";
      try {
        const response = await fetch("/api/tone");
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        $("audio").src = url;
        await $("audio").play();
        log("If you hear a beep, browser audio works.");
      } catch (error) {
        $("status").textContent = "Tone error";
        log(String(error.message || error));
      }
    }

    fillSelect("speaker", speakers, "Ryan");
    fillSelect("language", languages, "English");
    $("speak").addEventListener("click", speak);
    $("tone").addEventListener("click", testTone);
    refreshHealth();
  </script>
</body>
</html>
"""


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class TTSState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.model_path = None
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> Qwen3TTSModel:
        if self.model is not None:
            return self.model

        with self.lock:
            if self.model is not None:
                return self.model

            HF_HOME.mkdir(exist_ok=True)
            model_path = self.args.model
            if "/" in model_path and not Path(model_path).exists():
                print(f"Downloading/loading model snapshot: {model_path}", flush=True)
                model_path = snapshot_download(model_path, cache_dir=str(HF_HOME))

            print(f"Loading model from: {model_path}", flush=True)
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.args.device,
                dtype=dtype_from_name(self.args.dtype),
                attn_implementation="eager",
            )
            self.model_path = model_path
            print("Model loaded.", flush=True)
            return self.model

    def speak(self, payload: dict) -> tuple[bytes, dict]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Text is required.")

        speaker = str(payload.get("speaker") or "Ryan")
        language = str(payload.get("language") or "English")
        max_new_tokens = int(payload.get("max_new_tokens") or self.args.max_new_tokens)
        do_sample = bool(payload.get("do_sample", True))
        subtalker_dosample = bool(payload.get("subtalker_dosample", False))

        tts = self.load()
        started = time.perf_counter()
        with self.lock:
            wavs, sample_rate = tts.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                non_streaming_mode=True,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                subtalker_dosample=subtalker_dosample,
                top_p=0.8,
                temperature=0.8,
            )

        raw_wav = np.asarray(wavs[0])
        wav = normalize_audio(raw_wav)
        audio = io.BytesIO()
        sf.write(audio, wav, sample_rate, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - started
        metadata = {
            "sample_rate": sample_rate,
            "seconds": len(wav) / sample_rate,
            "elapsed": elapsed,
            "speaker": speaker,
            "language": language,
            "max_new_tokens": max_new_tokens,
            "raw_dtype": str(raw_wav.dtype),
            "raw_min": float(np.min(raw_wav)) if raw_wav.size else 0.0,
            "raw_max": float(np.max(raw_wav)) if raw_wav.size else 0.0,
            "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0,
        }
        return audio.getvalue(), metadata


class Handler(BaseHTTPRequestHandler):
    state: TTSState

    def _send_bytes(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if headers:
            for name, value in headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: dict) -> None:
        self._send_bytes(status, json.dumps(data).encode("utf-8"), "application/json")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html = INDEX_HTML.replace("__SPEAKERS__", json.dumps(SPEAKERS)).replace(
                "__LANGUAGES__", json.dumps(LANGUAGES)
            )
            self._send_bytes(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "model_loaded": self.state.loaded,
                    "model": self.state.args.model,
                    "device": self.state.args.device,
                    "dtype": self.state.args.dtype,
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
            )
            return

        if path == "/api/tone":
            self._send_bytes(HTTPStatus.OK, make_test_tone(), "audio/wav")
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/speak":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                payload = json.loads(raw.decode("utf-8") or "{}")
            else:
                payload = {key: values[0] for key, values in parse_qs(raw.decode("utf-8")).items()}

            audio, metadata = self.state.speak(payload)
            headers = {"X-Qwen3-TTS-Metadata": json.dumps(metadata)}
            self._send_bytes(HTTPStatus.OK, audio, "audio/wav", headers=headers)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "CUDA ran out of memory. Try less text or lower max_new_tokens.", "details": str(error)},
            )
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser server for Qwen3-TTS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--preload", action="store_true", help="Load the model before accepting requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = TTSState(args)

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU memory free/total: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GB", flush=True)

    if args.preload:
        state.load()

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
