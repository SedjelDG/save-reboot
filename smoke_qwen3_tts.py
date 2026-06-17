import argparse
import os
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Qwen3-TTS local smoke test.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument("--speaker", default="Ryan", choices=SPEAKERS)
    parser.add_argument("--language", default="English", choices=LANGUAGES)
    parser.add_argument(
        "--text",
        default=(
            "The rain had stopped, but the city still sounded like it was "
            "holding its breath."
        ),
    )
    parser.add_argument("--output", default="qwen3_tts_output.wav")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-sample", action="store_true", help="Disable both model sampling paths.")
    parser.add_argument("--no-talker-sample", action="store_true", help="Disable the main talker sampler.")
    parser.add_argument("--no-subtalker-sample", action="store_true", help="Disable the subtalker sampler.")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    HF_HOME.mkdir(exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"GPU memory free/total: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GB")

    model_path = args.model
    if "/" in args.model and not Path(args.model).exists():
        print(f"Downloading/loading model snapshot: {args.model}")
        model_path = snapshot_download(args.model, cache_dir=str(HF_HOME))

    print(f"Loading model from: {model_path}")
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=args.device,
        dtype=dtype_from_name(args.dtype),
        attn_implementation="eager",
    )

    print("Generating audio...")
    wavs, sample_rate = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        non_streaming_mode=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=not (args.no_sample or args.no_talker_sample),
        subtalker_dosample=not (args.no_sample or args.no_subtalker_sample),
        top_p=0.8,
        temperature=0.8,
    )

    output_path = ROOT / args.output
    sf.write(output_path, wavs[0], sample_rate)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print(
            "CUDA ran out of memory. Try a shorter --text, lower --max-new-tokens, "
            "or --device cpu --dtype float32.",
            file=sys.stderr,
        )
        raise
