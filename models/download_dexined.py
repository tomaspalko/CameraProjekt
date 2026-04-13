"""Download pre-trained DexiNed weights from Hugging Face Hub.

Usage:
    py -3.12 models/download_dexined.py

The script downloads the official BIPED-trained checkpoint (~15 MB) and saves it
to  models/dexined.pth  inside the project root.
"""
import os
import sys
import urllib.request
from pathlib import Path

# Official weights from Hugging Face (xavysp/dexined model repo)
URL = "https://huggingface.co/opencv/edge_detection_dexined/resolve/main/edge_detection_dexined_2024sep.onnx"

DEST = Path(__file__).parent / "dexined.onnx"


def _progress(count, block_size, total):
    pct = min(100.0, count * block_size / total * 100) if total > 0 else 0
    mb  = count * block_size / 1024 / 1024
    print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)


def main() -> None:
    if DEST.exists():
        print(f"Weights already present: {DEST}")
        return

    print(f"Downloading DexiNed weights from Hugging Face …")
    print(f"  Source : {URL}")
    print(f"  Dest   : {DEST}")

    try:
        urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
        print(f"\nDone!  Saved to {DEST}  ({DEST.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("\nFallback instructions:")
        print("  1. Open https://huggingface.co/xavysp/dexined/resolve/main/dexined.pth")
        print(f"  2. Save the file to: {DEST}")
        sys.exit(1)


if __name__ == "__main__":
    main()
