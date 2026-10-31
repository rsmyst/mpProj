"""Spoken-cue dispatcher.

The pose loop must never block on a subprocess, so cues go through a small
worker queue. A cooldown prevents the same corrective message from repeating
on every frame.
"""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

from reap.audio.tts import TTSBackend, create_tts


class CueEngine:
    def __init__(self, tts: TTSBackend | None = None,
                 min_gap_s: float = 1.2,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.tts = tts or create_tts()
        self.min_gap_s = float(min_gap_s)
        self._clock = clock
        self._last_at = float("-inf")
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        self.last_spoken = ""
        self._worker = threading.Thread(target=self._run, name="cue-voice",
                                        daemon=True)
        self._worker.start()

    def say(self, text: str, force: bool = False) -> bool:
        """Queue text for speech. Returns False when suppressed by cooldown."""
        text = " ".join(str(text).split())
        if not text or self._closed:
            return False
        now = self._clock()
        with self._lock:
            if not force and now - self._last_at < self.min_gap_s:
                return False
            self._last_at = now
            self.last_spoken = text
        self._queue.put(text)
        return True

    def say_now(self, text: str) -> None:
        """Speak synchronously (CLI tests and one-off announcements)."""
        text = " ".join(str(text).split())
        if not text:
            return
        with self._lock:
            self.last_spoken = text
            self._last_at = self._clock()
        try:
            self.tts.speak(text)
        except Exception as exc:  # noqa: BLE001 - voice must not kill the loop
            print(f"[audio] speech failed: {exc}")

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until queued speech finishes (used by tests)."""
        done = threading.Event()

        def _join() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=_join, daemon=True).start()
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                try:
                    self.tts.speak(text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[audio] speech failed: {exc}")
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        try:
            self.tts.close()
        except Exception:  # noqa: BLE001
            pass
