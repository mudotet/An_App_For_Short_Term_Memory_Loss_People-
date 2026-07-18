from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    started_at: datetime
    ended_at: datetime


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str = "vi",
        sample_rate: int = 16000,
        chunk_seconds: int = 5,
        enabled: bool = True,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._transcripts: "queue.Queue[TranscriptChunk]" = queue.Queue()
        self._errors: "queue.Queue[str]" = queue.Queue()

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="audio-transcriber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.chunk_seconds + 3)

    def pop_transcripts(self) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        while True:
            try:
                chunks.append(self._transcripts.get_nowait())
            except queue.Empty:
                return chunks

    def pop_errors(self) -> list[str]:
        errors: list[str] = []
        while True:
            try:
                errors.append(self._errors.get_nowait())
            except queue.Empty:
                return errors

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as exc:
            self._errors.put(f"Audio/STT is disabled: {exc}")
            return

        frame_count = int(self.sample_rate * self.chunk_seconds)
        while not self._stop.is_set():
            started_at = datetime.now()
            try:
                audio = sd.rec(
                    frame_count,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                if self._stop.is_set():
                    break
                text = self._transcribe_audio(model, sf, audio)
            except Exception as exc:
                self._errors.put(f"Could not record or transcribe audio: {exc}")
                time.sleep(1)
                continue

            text = " ".join(text.split())
            if text:
                self._transcripts.put(
                    TranscriptChunk(
                        text=text,
                        started_at=started_at,
                        ended_at=datetime.now(),
                    )
                )

    def _transcribe_audio(self, model: object, soundfile_module: object, audio: object) -> str:
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            soundfile_module.write(temp_path, audio, self.sample_rate)
            segments, _ = model.transcribe(
                temp_path,
                language=self.language,
                vad_filter=True,
                beam_size=1,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


class ReminderSpeaker:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: int = 90,
        rate: int = 170,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.rate = rate
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._errors: "queue.Queue[str]" = queue.Queue()
        self._last_spoken_by_key: dict[str, float] = {}

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._run, name="reminder-speaker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=3)

    def speak_once(self, key: str, message: str) -> None:
        if not self.enabled or not message.strip():
            return
        now = time.monotonic()
        last_spoken = self._last_spoken_by_key.get(key, 0)
        if now - last_spoken < self.cooldown_seconds:
            return
        self._last_spoken_by_key[key] = now
        self._queue.put(message.strip())

    def pop_errors(self) -> list[str]:
        errors: list[str] = []
        while True:
            try:
                errors.append(self._errors.get_nowait())
            except queue.Empty:
                return errors

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as exc:
            self._errors.put(f"TTS is disabled: {exc}")
            return

        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if message is None:
                break
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception as exc:
                self._errors.put(f"Could not speak reminder: {exc}")
