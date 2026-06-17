import argparse
import io
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent
TMP = ROOT / ".tmp"
HF_HOME = ROOT / ".hf-cache"
NEUTTS_ESPEAK = ROOT / ".venv-neutts" / "Lib" / "site-packages" / "neutts" / "espeak-ng.dll"
TORCH_LIB = ROOT / ".venv-neutts" / "Lib" / "site-packages" / "torch" / "lib"
SAMPLE_DIR = ROOT / "samples" / "neutts"
LEGACY_SAMPLE_DIR = ROOT / "neutts" / "samples"

os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(NEUTTS_ESPEAK))
os.environ["PATH"] = f"{TORCH_LIB}{os.pathsep}{os.environ.get('PATH', '')}"
if hasattr(os, "add_dll_directory") and TORCH_LIB.exists():
    os.add_dll_directory(str(TORCH_LIB))

from neutts import NeuTTS  # noqa: E402


VOICES = ["jo", "dave", "greta", "juliette", "mateo"]


def make_test_tone(sample_rate: int = 24000, seconds: float = 1.0, frequency: float = 440.0) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    wav = (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    audio = io.BytesIO()
    sf.write(audio, wav, sample_rate, format="WAV", subtype="PCM_16")
    return audio.getvalue()


class NeuTTSState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tts: NeuTTS | None = None
        self.refs: dict[str, tuple[torch.Tensor, str]] = {}
        self.load_lock = threading.Lock()
        self.infer_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.tts is not None

    def get_tts(self) -> NeuTTS:
        if self.tts is not None:
            return self.tts

        with self.load_lock:
            if self.tts is not None:
                return self.tts

            print(
                "Loading NeuTTS "
                f"backbone={self.args.backbone} backbone_device={self.args.backbone_device} "
                f"codec={self.args.codec} codec_device={self.args.codec_device}",
                flush=True,
            )
            started = time.perf_counter()
            self.tts = NeuTTS(
                backbone_repo=self.args.backbone,
                backbone_device=self.args.backbone_device,
                codec_repo=self.args.codec,
                codec_device=self.args.codec_device,
            )
            print(f"NeuTTS loaded in {time.perf_counter() - started:.2f}s.", flush=True)
            return self.tts

    def get_reference(self, voice: str) -> tuple[torch.Tensor, str]:
        if voice not in VOICES:
            raise ValueError(f"Unsupported NeuTTS voice: {voice}")

        if voice in self.refs:
            return self.refs[voice]

        sample_dir = SAMPLE_DIR if SAMPLE_DIR.exists() else LEGACY_SAMPLE_DIR
        codes_path = sample_dir / f"{voice}.pt"
        text_path = sample_dir / f"{voice}.txt"
        if not codes_path.exists() or not text_path.exists():
            raise FileNotFoundError(f"Missing reference files for voice '{voice}'.")

        ref_codes = torch.load(codes_path, map_location="cpu")
        ref_text = text_path.read_text(encoding="utf-8").strip()
        self.refs[voice] = (ref_codes, ref_text)
        return self.refs[voice]

    def speak(self, payload: dict) -> tuple[bytes, dict]:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Text is required.")

        voice = str(payload.get("voice") or payload.get("speaker") or self.args.voice)
        max_chars = max(80, min(900, int(payload.get("max_chars") or self.args.max_chars)))
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] or text[:max_chars]

        tts = self.get_tts()
        ref_codes, ref_text = self.get_reference(voice)

        started = time.perf_counter()
        with self.infer_lock:
            wav = tts.infer(text, ref_codes, ref_text)

        wav = np.asarray(wav, dtype=np.float32)
        audio = io.BytesIO()
        sf.write(audio, wav, 24000, format="WAV", subtype="PCM_16")
        elapsed = time.perf_counter() - started
        metadata = {
            "sample_rate": 24000,
            "seconds": len(wav) / 24000,
            "elapsed": elapsed,
            "voice": voice,
            "engine": "neutts",
            "backbone": self.args.backbone,
            "backbone_device": self.args.backbone_device,
            "codec_device": self.args.codec_device,
            "chars": len(text),
            "peak": float(np.max(np.abs(wav))) if wav.size else 0.0,
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) if wav.size else 0.0,
        }
        return audio.getvalue(), metadata


class Handler(BaseHTTPRequestHandler):
    state: NeuTTSState

    def _send_bytes(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-TTS-Metadata")
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
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "engine": "neutts",
                    "pipeline_loaded": self.state.loaded,
                    "voices": VOICES,
                    "backbone": self.state.args.backbone,
                    "backbone_device": self.state.args.backbone_device,
                    "codec": self.state.args.codec,
                    "codec_device": self.state.args.codec_device,
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
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser server for NeuTTS Air.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--backbone", default="neuphonic/neutts-air-q4-gguf")
    parser.add_argument("--codec", default="neuphonic/neucodec-onnx-decoder")
    parser.add_argument("--backbone-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--voice", default="jo", choices=VOICES)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--preload", action="store_true", help="Load NeuTTS before accepting requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TMP.mkdir(exist_ok=True)
    HF_HOME.mkdir(exist_ok=True)
    state = NeuTTSState(args)

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.preload:
        state.get_tts()

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NeuTTS server listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
