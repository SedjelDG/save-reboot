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
TORCH_LIB = ROOT / ".venv-neutts" / "Lib" / "site-packages" / "torch" / "lib"

os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(NEUTTS_ESPEAK))
os.environ["PATH"] = f"{TORCH_LIB}{os.pathsep}{os.environ.get('PATH', '')}"
if hasattr(os, "add_dll_directory") and TORCH_LIB.exists():
    os.add_dll_directory(str(TORCH_LIB))

from neutts import NeuTTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuTTS Nano Q4 smoke test.")
    parser.add_argument("--text", default="This is NeuTTS Nano voice cloning running locally.")
    parser.add_argument("--output", default="neutts_air_output.wav")
    parser.add_argument("--backbone", default="neuphonic/neutts-nano-q4-gguf")
    parser.add_argument("--codec", default="neuphonic/neucodec-onnx-decoder")
    parser.add_argument("--backbone-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--codec-device", default="cpu")
    parser.add_argument("--ref-codes", default="samples/neutts/jo.pt")
    parser.add_argument("--ref-text", default="samples/neutts/jo.txt")
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TMP.mkdir(exist_ok=True)
    HF_HOME.mkdir(exist_ok=True)

    ref_codes_path = ROOT / args.ref_codes
    ref_text_path = ROOT / args.ref_text
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    ref_codes = torch.load(ref_codes_path, map_location="cpu")

    started = time.perf_counter()
    tts = NeuTTS(
        backbone_repo=args.backbone,
        backbone_device=args.backbone_device,
        codec_repo=args.codec,
        codec_device=args.codec_device,
    )
    load_elapsed = time.perf_counter() - started

    print(f"Loaded in {load_elapsed:.2f}s")
    for index in range(max(1, args.repeat)):
        infer_started = time.perf_counter()
        wav = tts.infer(args.text, ref_codes, ref_text)
        infer_elapsed = time.perf_counter() - infer_started

        output_path = ROOT / args.output
        if args.repeat > 1:
            output_path = output_path.with_stem(f"{output_path.stem}_{index + 1}")
        sf.write(output_path, wav, 24000, subtype="PCM_16")
        wav_np = np.asarray(wav)
        print(f"Run {index + 1}: generated {len(wav_np) / 24000:.2f}s of audio in {infer_elapsed:.2f}s")
        print(f"Run {index + 1}: peak {float(np.max(np.abs(wav_np))) if wav_np.size else 0.0:.4f}")
        print(f"Run {index + 1}: saved {output_path}")


if __name__ == "__main__":
    main()
