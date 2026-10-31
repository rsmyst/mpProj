"""Local text-to-speech backends for spoken coaching cues.

Priority is Phonon-31M (Gradium's 31M-parameter streaming TTS), then Piper,
then the system espeak-ng. All three run fully on-device: no cloud calls during
a set.

Phonon loads ``hf://gradium/phonon-31m-en`` through the ``ptts`` runtime. The
q8 weights are only ~64 MB and are cached after the first run. Pre-fetch them
with ``uv run python scripts/download_phonon.py``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np


class TTSBackend(Protocol):
    name: str

    def speak(self, text: str) -> None: ...
    def close(self) -> None: ...


class NullTTS:
    """Silent backend that records text (tests, headless machines)."""

    name = "null"

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def close(self) -> None:
        pass


def _which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _venv_bin(name: str) -> str | None:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else None


def _pick_player() -> str | None:
    # paplay (PipeWire/PulseAudio) is much more reliable than raw ALSA aplay.
    return _which("paplay", "aplay", "ffplay")


def _play_wav(player: str, wav: str, timeout: float = 60.0) -> None:
    name = Path(player).name
    if name == "paplay":
        cmd = [player, wav]
    elif name == "aplay":
        cmd = [player, "-q", wav]
    else:
        cmd = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", wav]
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[audio] audio playback timed out (no output device?)")


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        wav = fh.name
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return wav


class PhononTTS:
    """Natural on-device voice via Gradium Phonon-31M (q8 GGUF)."""

    name = "phonon"

    def __init__(self, model: str = "hf://gradium/phonon-31m-en",
                 voice: str | None = "Pippa", lang: str = "en",
                 quant: str = "q8", device: str = "cpu",
                 voice_catalog: str | None = None) -> None:
        try:
            import ptts
        except ImportError as exc:
            raise RuntimeError(
                "ptts is not installed; run `uv sync --extra voice`"
            ) from exc

        self.player = _pick_player()
        if not self.player:
            raise RuntimeError("no audio player found (paplay/aplay/ffplay)")
        self.model = model
        self.voice = voice or None
        self.lang = lang
        self.quant = quant
        kwargs = {
            "model": model,
            "lang": lang,
            "quant": quant,
            "device": device,
            "voice": self.voice,
        }
        if voice_catalog:
            kwargs["voice_catalog"] = voice_catalog
        self._tts = ptts.TTS(**kwargs)

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            wav = fh.name
        try:
            self._tts.save(wav, text, voice=self.voice)
            _play_wav(self.player, wav)
        finally:
            Path(wav).unlink(missing_ok=True)

    def close(self) -> None:
        self._tts = None


class EspeakTTS:
    """Very small, dependency-free voice using the system espeak-ng binary."""

    name = "espeak"

    def __init__(self, voice: str = "en", speed: int = 165,
                 amplitude: int = 120) -> None:
        self.bin = _which("espeak-ng", "espeak")
        if not self.bin:
            raise RuntimeError("espeak-ng/espeak not found on this system")
        self.voice = voice
        self.speed = int(speed)
        self.amplitude = int(amplitude)
        self.player = _pick_player()

    def speak(self, text: str) -> None:
        base = [self.bin, "-v", self.voice, "-s", str(self.speed),
                "-a", str(self.amplitude)]
        if not self.player:
            subprocess.run(base + [text], check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            wav = fh.name
        try:
            subprocess.run(base + ["-w", wav, text], check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
            _play_wav(self.player, wav)
        finally:
            Path(wav).unlink(missing_ok=True)

    def close(self) -> None:
        pass


class PiperTTS:
    """Natural offline voice through the ``piper`` CLI + a local .onnx voice."""

    name = "piper"

    def __init__(self, model_path: str | os.PathLike[str] | None = None,
                 length_scale: float = 1.0) -> None:
        self.bin = _which("piper") or _venv_bin("piper")
        if not self.bin:
            raise RuntimeError("piper CLI not installed (uv sync --extra voice)")
        model = model_path or os.environ.get("REAP_PIPER_MODEL")
        if not model:
            default = Path("models/en_US-amy-low.onnx")
            model = str(default) if default.exists() else None
        if not model or not Path(model).exists():
            raise FileNotFoundError(
                "Piper voice model not found; run scripts/download_piper_voice.py "
                "or set REAP_PIPER_MODEL"
            )
        self.model = str(model)
        self.length_scale = float(length_scale)
        self.player = _pick_player()
        if not self.player:
            raise RuntimeError("no audio player found (paplay/aplay/ffplay)")

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            wav = fh.name
        try:
            subprocess.run(
                [self.bin, "-m", self.model, "--length-scale",
                 f"{self.length_scale:.2f}", "-f", wav],
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            _play_wav(self.player, wav)
        finally:
            Path(wav).unlink(missing_ok=True)

    def close(self) -> None:
        pass


def create_tts(prefer: str = "auto", *,
               phonon_model: str = "hf://gradium/phonon-31m-en",
               phonon_voice: str | None = "Pippa",
               phonon_lang: str = "en",
               phonon_quant: str = "q8",
               phonon_device: str = "cpu",
               phonon_voice_catalog: str | None = None,
               piper_model: str | None = None,
               length_scale: float = 1.0,
               espeak_voice: str = "en",
               espeak_speed: int = 165,
               espeak_amplitude: int = 120) -> TTSBackend:
    """Build the best available local voice backend.

    prefer: auto | phonon | piper | espeak | none
    """
    prefer = (prefer or "auto").lower()
    if prefer == "none":
        return NullTTS()

    if prefer in ("auto", "phonon"):
        try:
            tts = PhononTTS(
                model=phonon_model, voice=phonon_voice, lang=phonon_lang,
                quant=phonon_quant, device=phonon_device,
                voice_catalog=phonon_voice_catalog,
            )
            print(f"[audio] using Phonon TTS ({tts.model}, quant={tts.quant})")
            return tts
        except Exception as exc:  # noqa: BLE001 - voice is optional
            if prefer == "phonon":
                print(f"[audio] Phonon unavailable ({exc}); trying Piper")

    if prefer in ("auto", "phonon", "piper"):
        try:
            tts = PiperTTS(model_path=piper_model, length_scale=length_scale)
            print(f"[audio] using Piper TTS ({Path(tts.model).name})")
            return tts
        except Exception as exc:  # noqa: BLE001 - voice is optional
            if prefer in ("piper", "phonon"):
                print(f"[audio] Piper unavailable ({exc}); trying espeak")

    if prefer in ("auto", "phonon", "piper", "espeak"):
        try:
            tts = EspeakTTS(voice=espeak_voice, speed=espeak_speed,
                            amplitude=espeak_amplitude)
            print("[audio] using espeak-ng TTS")
            return tts
        except RuntimeError as exc:
            if prefer == "espeak":
                print(f"[audio] espeak unavailable ({exc}); using silent TTS")

    print("[audio] no TTS backend available; using silent TTS")
    return NullTTS()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Speak text with the local REAP voice")
    ap.add_argument("text", nargs="+", help="words to speak")
    ap.add_argument("--tts", default="auto",
                    choices=["auto", "phonon", "piper", "espeak", "none"])
    ap.add_argument("--voice", default="Pippa", help="Phonon voice name/ID (default: Pippa)")
    ap.add_argument("--quant", default="q8", help="Phonon weight format (q8, q6k, q4k, f32)")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args(argv)
    tts = create_tts(args.tts, phonon_voice=args.voice,
                     phonon_quant=args.quant, phonon_lang=args.lang)
    try:
        tts.speak(" ".join(args.text))
    finally:
        tts.close()
    return 0
