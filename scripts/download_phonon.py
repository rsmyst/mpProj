"""Pre-fetch Gradium Phonon-31M (q8) so TTS works fully offline."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Phonon-31M TTS weights")
    ap.add_argument("--model", default="hf://gradium/phonon-31m-en")
    ap.add_argument("--voice", default="Pippa",
                    help="catalog voice name/ID (default: Pippa, British female)")
    ap.add_argument("--quant", default="q8")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args(argv)

    try:
        import ptts
    except ImportError:
        print("error: ptts not installed; run: uv sync --extra voice",
              file=sys.stderr)
        return 2

    print(f"[download] loading {args.model} (quant={args.quant}) ...")
    t0 = time.time()
    try:
        tts = ptts.TTS(model=args.model, lang=args.lang, quant=args.quant,
                       device="cpu", voice=args.voice)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(tempfile.mktemp(suffix=".wav"))
    try:
        duration = tts.save(out, "Ready.", voice=args.voice)
        print(f"[download] done in {time.time() - t0:.1f}s "
              f"(test clip {duration:.2f}s, {out.stat().st_size} bytes)")
    finally:
        out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
