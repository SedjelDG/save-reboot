import argparse
import os
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Kokoro TTS smoke test.")
    parser.add_argument("--text", default="Hello. This is Kokoro running locally.")
    parser.add_argument("--voice", default="af_heart", choices=VOICES)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", default="kokoro_output.wav")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HF_HOME.mkdir(exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    started = time.perf_counter()
    pipeline = KPipeline(lang_code="a", device=args.device)
    load_elapsed = time.perf_counter() - started
    print(f"Loaded pipeline in {load_elapsed:.2f}s")

    chunks = []
    gen_started = time.perf_counter()
    for result in pipeline(args.text, voice=args.voice, speed=args.speed):
        chunks.append(np.asarray(result.audio, dtype=np.float32))

    if not chunks:
        raise RuntimeError("Kokoro returned no audio.")

    wav = np.concatenate(chunks)
    output_path = ROOT / args.output
    sf.write(output_path, wav, 24000, format="WAV", subtype="PCM_16")
    gen_elapsed = time.perf_counter() - gen_started
    duration = len(wav) / 24000
    print(f"Generated {duration:.2f}s of audio in {gen_elapsed:.2f}s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
