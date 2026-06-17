import argparse
import base64
import io
import json
import os
import re
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


BUILTIN_VOICES = ["jo", "dave", "greta", "juliette", "mateo"]
CUSTOM_SAMPLE_DIR = SAMPLE_DIR / "custom"


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
        self.encoder_codec = None
        self.refs: dict[str, tuple[torch.Tensor, str]] = {}
        self.load_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.encode_lock = threading.Lock()

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

    def voice_dirs(self) -> list[Path]:
        dirs = []
        if SAMPLE_DIR.exists():
            dirs.append(SAMPLE_DIR)
        if CUSTOM_SAMPLE_DIR.exists():
            dirs.append(CUSTOM_SAMPLE_DIR)
        if LEGACY_SAMPLE_DIR.exists():
            dirs.append(LEGACY_SAMPLE_DIR)
        return dirs

    def list_voices(self) -> list[dict]:
        seen = set()
        voices = []
        for directory in self.voice_dirs():
            kind = "custom" if directory == CUSTOM_SAMPLE_DIR else "preset"
            for codes_path in sorted(directory.glob("*.pt")):
                text_path = codes_path.with_suffix(".txt")
                if not text_path.exists() or codes_path.stem in seen:
                    continue
                seen.add(codes_path.stem)
                voices.append({"id": codes_path.stem, "label": codes_path.stem, "kind": kind})
        return voices

    def find_reference_paths(self, voice: str) -> tuple[Path, Path] | None:
        for directory in self.voice_dirs():
            codes_path = directory / f"{voice}.pt"
            text_path = directory / f"{voice}.txt"
            if codes_path.exists() and text_path.exists():
                return codes_path, text_path
        return None

    def get_reference(self, voice: str) -> tuple[torch.Tensor, str]:
        paths = self.find_reference_paths(voice)
        if not paths:
            raise ValueError(f"Unsupported NeuTTS voice: {voice}")

        if voice in self.refs:
            return self.refs[voice]

        codes_path, text_path = paths
        ref_codes = torch.load(codes_path, map_location="cpu")
        if ref_codes.numel() > self.args.max_reference_codes:
            ref_codes = ref_codes.flatten()[: self.args.max_reference_codes]
        ref_text = text_path.read_text(encoding="utf-8").strip()
        self.refs[voice] = (ref_codes, ref_text)
        return self.refs[voice]

    def get_encoder_codec(self):
        if self.encoder_codec is not None:
            return self.encoder_codec

        with self.load_lock:
            if self.encoder_codec is not None:
                return self.encoder_codec

            from neucodec import NeuCodec

            print(f"Loading NeuCodec encoder on {self.args.encoder_device} ...", flush=True)
            started = time.perf_counter()
            codec = NeuCodec.from_pretrained("neuphonic/neucodec")
            codec.eval().to(self.args.encoder_device)
            self.encoder_codec = codec
            print(f"NeuCodec encoder loaded in {time.perf_counter() - started:.2f}s.", flush=True)
            return self.encoder_codec

    def sanitize_voice_name(self, name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
        if not clean:
            raise ValueError("Voice name is required.")
        if clean in {"con", "prn", "aux", "nul"}:
            raise ValueError("That voice name is reserved on Windows.")
        return clean[:48]

    def create_voice(self, payload: dict) -> dict:
        name = self.sanitize_voice_name(str(payload.get("name", "")))
        transcript = str(payload.get("transcript", "")).strip()
        audio_base64 = str(payload.get("audio_base64", "")).strip()
        if not transcript:
            raise ValueError("Transcript is required for NeuTTS reference cloning.")
        if not audio_base64:
            raise ValueError("Audio data is required.")
        if "," in audio_base64 and audio_base64.split(",", 1)[0].startswith("data:"):
            audio_base64 = audio_base64.split(",", 1)[1]

        CUSTOM_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = CUSTOM_SAMPLE_DIR / f"{name}.wav"
        codes_path = CUSTOM_SAMPLE_DIR / f"{name}.pt"
        text_path = CUSTOM_SAMPLE_DIR / f"{name}.txt"

        audio_bytes = base64.b64decode(audio_base64)
        if len(audio_bytes) < 1024:
            raise ValueError("Audio sample is too small.")
        if len(audio_bytes) > self.args.max_reference_mb * 1024 * 1024:
            raise ValueError(f"Audio sample is larger than {self.args.max_reference_mb} MB.")

        audio_path.write_bytes(audio_bytes)
        text_path.write_text(transcript, encoding="utf-8")

        with self.encode_lock:
            from librosa import load

            codec = self.get_encoder_codec()
            wav, _ = load(audio_path, sr=16000, mono=True)
            max_samples = int(self.args.max_reference_seconds * 16000)
            trimmed = False
            if len(wav) > max_samples:
                wav = wav[:max_samples]
                trimmed = True
            sf.write(audio_path, wav, 16000, format="WAV", subtype="PCM_16")
            wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            if ref_codes.numel() > self.args.max_reference_codes:
                ref_codes = ref_codes.flatten()[: self.args.max_reference_codes]
                trimmed = True
            torch.save(ref_codes.cpu(), codes_path)

        self.refs.pop(name, None)
        return {
            "id": name,
            "label": name,
            "kind": "custom",
            "audio_path": str(audio_path.relative_to(ROOT)),
            "codes_path": str(codes_path.relative_to(ROOT)),
            "text_path": str(text_path.relative_to(ROOT)),
            "seconds": len(wav) / 16000,
            "codes": int(ref_codes.numel()),
            "trimmed": trimmed,
        }

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
            "reference_codes": int(ref_codes.numel()),
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
                    "voices": self.state.list_voices(),
                    "backbone": self.state.args.backbone,
                    "backbone_device": self.state.args.backbone_device,
                    "codec": self.state.args.codec,
                    "codec_device": self.state.args.codec_device,
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
            )
            return

        if path == "/api/voices":
            self._send_json(HTTPStatus.OK, {"voices": self.state.list_voices()})
            return

        if path == "/api/tone":
            self._send_bytes(HTTPStatus.OK, make_test_tone(), "audio/wav")
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/speak", "/api/voices"}:
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

            if path == "/api/voices":
                voice = self.state.create_voice(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "voice": voice, "voices": self.state.list_voices()})
            else:
                audio, metadata = self.state.speak(payload)
                headers = {"X-TTS-Metadata": json.dumps(metadata)}
                self._send_bytes(HTTPStatus.OK, audio, "audio/wav", headers=headers)
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser server for NeuTTS Nano.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--backbone", default="neuphonic/neutts-nano-q4-gguf")
    parser.add_argument("--codec", default="neuphonic/neucodec-onnx-decoder")
    parser.add_argument("--backbone-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--voice", default="jo")
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--max-reference-mb", type=int, default=30)
    parser.add_argument("--max-reference-seconds", type=float, default=12.0)
    parser.add_argument("--max-reference-codes", type=int, default=650)
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
