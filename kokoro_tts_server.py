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
from kokoro import KPipeline


ROOT = Path(__file__).resolve().parent
HF_HOME = ROOT / ".hf-cache"
os.environ.setdefault("HF_HOME", str(HF_HOME))

VOICES = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]

LANGUAGES = [
    ("a", "American English"),
    ("b", "British English"),
]

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Kokoro Local Reader Test</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #202522;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: #f7f7f4;
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
      border: 1px solid #c9d0ca;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
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
      border: 1px solid #c9d0ca;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.78);
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

    .secondary { background: #626c66; }

    audio {
      width: 100%;
      margin-top: 16px;
    }

    pre {
      min-height: 112px;
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

      .status { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Kokoro Local Reader Test</h1>
      <div class="status" id="status">Checking server...</div>
    </header>

    <div class="layout">
      <section>
        <textarea id="text">Hello. This is Kokoro running locally.</textarea>
        <audio id="audio" controls></audio>
      </section>

      <aside>
        <label>
          Voice
          <select id="voice"></select>
        </label>
        <label>
          Language
          <select id="language"></select>
        </label>
        <label>
          Speed
          <input id="speed" type="number" min="0.6" max="1.5" step="0.05" value="1.0" />
        </label>
        <button id="speak">Generate & Play</button>
        <button id="tone" type="button" class="secondary">Test Tone</button>
        <pre id="log"></pre>
      </aside>
    </div>
  </main>

  <script>
    const voices = __VOICES__;
    const languages = __LANGUAGES__;
    const $ = (id) => document.getElementById(id);

    function fillSelect(id, values, selected) {
      const select = $(id);
      select.innerHTML = values.map((item) => {
        const value = Array.isArray(item) ? item[0] : item;
        const label = Array.isArray(item) ? item[1] : item;
        const attr = value === selected ? " selected" : "";
        return `<option value="${value}"${attr}>${label}</option>`;
      }).join("");
    }

    function log(message) {
      $("log").textContent = message;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        $("status").textContent = data.pipeline_loaded ? "Pipeline loaded" : "Ready";
      } catch (error) {
        $("status").textContent = "Server unavailable";
      }
    }

    async function speak() {
      const button = $("speak");
      button.disabled = true;
      $("status").textContent = "Generating...";
      log("Sending request.");

      const started = performance.now();
      try {
        const response = await fetch("/api/speak", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: $("text").value,
            voice: $("voice").value,
            lang_code: $("language").value,
            speed: Number($("speed").value)
          })
        });

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const elapsed = ((performance.now() - started) / 1000).toFixed(1);
        const metadata = response.headers.get("X-TTS-Metadata");
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

    fillSelect("voice", voices, "af_heart");
    fillSelect("language", languages, "a");
    $("speak").addEventListener("click", speak);
    $("tone").addEventListener("click", testTone);
    refreshHealth();
  </script>
</body>
</html>
"""


def make_test_tone(sample_rate=24000, seconds=1.0, frequency=440.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    wav = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    audio = io.BytesIO()
    sf.write(audio, wav, sample_rate, format="WAV", subtype="PCM_16")
    return audio.getvalue()


class KokoroState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pipelines: dict[str, KPipeline] = {}
        self.lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self.pipelines)

    def get_pipeline(self, lang_code: str) -> KPipeline:
        if lang_code not in {"a", "b"}:
            raise ValueError("Only American English (a) and British English (b) are configured.")

        if lang_code in self.pipelines:
            return self.pipelines[lang_code]

        with self.lock:
            if lang_code in self.pipelines:
                return self.pipelines[lang_code]

            print(f"Loading Kokoro pipeline lang={lang_code} device={self.args.device}", flush=True)
            pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=self.args.device)
            self.pipelines[lang_code] = pipeline
            print("Kokoro pipeline loaded.", flush=True)
            return pipeline

    def speak(self, payload: dict) -> tuple[bytes, dict]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Text is required.")

        voice = str(payload.get("voice") or "af_heart")
        if voice not in VOICES:
            raise ValueError(f"Unsupported voice: {voice}")

        lang_code = str(payload.get("lang_code") or self.args.lang_code)
        speed = float(payload.get("speed") or self.args.speed)
        speed = max(0.6, min(1.5, speed))

        pipeline = self.get_pipeline(lang_code)
        started = time.perf_counter()
        chunks = []
        with self.lock:
            for result in pipeline(text, voice=voice, speed=speed):
                chunks.append(np.asarray(result.audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        wav = np.concatenate(chunks)
        audio = io.BytesIO()
        sf.write(audio, wav, 24000, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - started
        metadata = {
            "sample_rate": 24000,
            "seconds": len(wav) / 24000,
            "elapsed": elapsed,
            "voice": voice,
            "lang_code": lang_code,
            "speed": speed,
            "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0,
        }
        return audio.getvalue(), metadata


class Handler(BaseHTTPRequestHandler):
    state: KokoroState

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
            html_path = ROOT / "kokoro_reader.html"
            html = html_path.read_text(encoding="utf-8") if html_path.exists() else INDEX_HTML
            html = html.replace("__VOICES__", json.dumps(VOICES)).replace("__LANGUAGES__", json.dumps(LANGUAGES))
            self._send_bytes(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "engine": "kokoro",
                    "pipeline_loaded": self.state.loaded,
                    "device": self.state.args.device,
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
            headers = {"X-TTS-Metadata": json.dumps(metadata)}
            self._send_bytes(HTTPStatus.OK, audio, "audio/wav", headers=headers)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "CUDA ran out of memory. Try less text or --device cpu.", "details": str(error)},
            )
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser server for Kokoro TTS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lang-code", default="a", choices=["a", "b"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--preload", action="store_true", help="Load the default pipeline before accepting requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HF_HOME.mkdir(exist_ok=True)
    state = KokoroState(args)

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.preload:
        state.get_pipeline(args.lang_code)

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
