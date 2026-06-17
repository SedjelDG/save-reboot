import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent
TMP = ROOT / ".tmp"
HF_HOME = ROOT / ".hf-cache"
NEUTTS_ESPEAK = ROOT / ".venv-neutts" / "Lib" / "site-packages" / "neutts" / "espeak-ng.dll"

os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(NEUTTS_ESPEAK))

from neutts import NeuTTS  # noqa: E402


SPEAKERS = ["jo", "dave", "greta", "juliette", "mateo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NeuTTS Nano samples for bundled voices.")
    parser.add_argument(
        "--text",
        default=(
            "The rain had stopped, but the city still sounded like it was holding its breath. "
            "She opened the old book and felt the room go silent."
        ),
    )
    parser.add_argument("--speaker", choices=SPEAKERS + ["all"], default="all")
    parser.add_argument("--output-dir", default="neutts_samples")
    parser.add_argument("--backbone", default="neuphonic/neutts-nano-q4-gguf")
    parser.add_argument("--codec", default="neuphonic/neucodec-onnx-decoder")
    parser.add_argument("--backbone-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--codec-device", default="cpu")
    return parser.parse_args()


def load_reference(name: str):
    ref_dir = ROOT / "samples" / "neutts"
    ref_codes = torch.load(ref_dir / f"{name}.pt", map_location="cpu")
    ref_text = (ref_dir / f"{name}.txt").read_text(encoding="utf-8").strip()
    return ref_codes, ref_text


def main() -> None:
    args = parse_args()
    TMP.mkdir(exist_ok=True)
    HF_HOME.mkdir(exist_ok=True)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    speakers = SPEAKERS if args.speaker == "all" else [args.speaker]

    load_started = time.perf_counter()
    tts = NeuTTS(
        backbone_repo=args.backbone,
        backbone_device=args.backbone_device,
        codec_repo=args.codec,
        codec_device=args.codec_device,
    )
    print(f"Loaded NeuTTS in {time.perf_counter() - load_started:.2f}s")

    for name in speakers:
        ref_codes, ref_text = load_reference(name)
        started = time.perf_counter()
        wav = tts.infer(args.text, ref_codes, ref_text)
        elapsed = time.perf_counter() - started
        wav_np = np.asarray(wav)
        output_path = output_dir / f"{name}.wav"
        sf.write(output_path, wav_np, 24000, subtype="PCM_16")
        print(
            f"{name}: {len(wav_np) / 24000:.2f}s audio in {elapsed:.2f}s, "
            f"peak={float(np.max(np.abs(wav_np))) if wav_np.size else 0.0:.3f}, "
            f"saved={output_path}"
        )


if __name__ == "__main__":
    main()
