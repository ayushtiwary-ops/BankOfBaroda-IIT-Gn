#!/usr/bin/env python3
"""Generate the scan-and-attack QR for the slide.

    python deploy/make_qr.py https://pramaan.fly.dev   # -> deploy/demo_qr.png
"""
import sys
from pathlib import Path


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://pramaan.fly.dev"
    out = Path(__file__).resolve().parent / "demo_qr.png"
    try:
        import qrcode
    except ImportError:
        print("pip install qrcode[pil] to generate the PNG, or use any QR tool for:")
        print(" ", url)
        return 0
    img = qrcode.make(url)
    img.save(out)
    print(f"wrote {out} for {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
