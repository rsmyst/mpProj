"""Pre-fetch a Piper voice so spoken cues work offline (uv sync --extra voice)."""
from pathlib import Path

VOICE = "en_US-amy-low"


def main() -> None:
    try:
        from piper.download_voices import download_voice
    except ImportError:
        raise SystemExit("piper-tts is not installed; run: uv sync --extra voice")

    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[download] fetching Piper voice {VOICE} -> {out}/ ...")
    download_voice(VOICE, out)
    print(f"[download] done: {out / (VOICE + '.onnx')}")


if __name__ == "__main__":
    main()
