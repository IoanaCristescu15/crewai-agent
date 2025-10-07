from __future__ import annotations

import contextlib
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from queue import Queue


@dataclass
class VoiceIO:
    """Utility class that handles audio capture, speech-to-text, and text-to-speech."""

    stt_model_name: str = "base"
    sample_rate: int = 16_000
    channels: int = 1
    tts_voice: Optional[str] = None
    tts_rate: Optional[int] = None

    _stt_model: any = None  # Lazy-loaded Whisper model
    _tts_engine: any = None  # Lazy-loaded pyttsx3 engine

    def capture_audio(self, output_path: Optional[Path] = None) -> Path:
        """Record microphone input until the user presses Enter again."""
        sounddevice = self._lazy_import_sounddevice()
        soundfile = self._lazy_import_soundfile()

        if output_path:
            path = Path(output_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                path = Path(tmp_file.name)

        audio_queue: Queue = Queue()

        def callback(indata, frames, time, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            audio_queue.put(indata.copy())

        input("Press Enter to start recording...")
        print("Recording... press Enter to stop.")

        try:
            with sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
            ):
                try:
                    input()
                except KeyboardInterrupt:
                    print("\nRecording interrupted by user.")
        except Exception as exc:
            raise RuntimeError(f"Could not access microphone: {exc}") from exc

        frames = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())

        if not frames:
            raise RuntimeError("No audio captured; try speaking closer to the microphone.")

        audio = np.concatenate(frames, axis=0)
        soundfile.write(path, audio, self.sample_rate)
        return path

    def capture_and_transcribe(self) -> Tuple[str, Path]:
        """Shortcut to record audio and immediately run STT."""
        audio_path = self.capture_audio()
        transcript = self.transcribe_file(audio_path)
        return transcript, audio_path

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an existing audio file with Whisper."""
        whisper = self._lazy_import_whisper()
        model = self._get_stt_model(whisper)
        result = model.transcribe(str(audio_path))
        return (result.get("text") or "").strip()

    def synthesize_speech(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Generate speech audio from text and return the saved file path."""
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        engine = self._get_tts_engine()
        if self.tts_voice:
            with contextlib.suppress(Exception):
                engine.setProperty("voice", self.tts_voice)
        if self.tts_rate:
            with contextlib.suppress(Exception):
                engine.setProperty("rate", self.tts_rate)

        if output_path:
            path = Path(output_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                path = Path(tmp_file.name)
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path

    def speak(self, text: str) -> None:
        """Speak text directly via the TTS engine (no file I/O)."""
        if not isinstance(text, str) or not text.strip():
            return
        engine = self._get_tts_engine()
        # Apply optional voice/rate if provided
        if self.tts_voice:
            with contextlib.suppress(Exception):
                engine.setProperty("voice", self.tts_voice)
        if self.tts_rate:
            with contextlib.suppress(Exception):
                engine.setProperty("rate", self.tts_rate)
        with contextlib.suppress(Exception):
            engine.stop()
        engine.say(text)
        engine.runAndWait()

    def play_audio(self, audio_path: str | Path) -> None:
        """Play an audio file through the default output device."""
        sounddevice = self._lazy_import_sounddevice()
        soundfile = self._lazy_import_soundfile()

        data, samplerate = soundfile.read(str(audio_path))
        sounddevice.play(data, samplerate)
        sounddevice.wait()

    def _get_stt_model(self, whisper_module):
        if self._stt_model is None:
            try:
                self._stt_model = whisper_module.load_model(self.stt_model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Whisper model '{self.stt_model_name}': {exc}"
                ) from exc
        return self._stt_model

    def _get_tts_engine(self):
        if self._tts_engine is None:
            pyttsx3 = self._lazy_import_pyttsx3()
            try:
                self._tts_engine = pyttsx3.init()
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize text-to-speech: {exc}") from exc
        return self._tts_engine

    @staticmethod
    def _lazy_import_whisper():
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'whisper' package is required for speech-to-text. "
                "Install it with `pip install openai-whisper`."
            ) from exc
        return whisper

    @staticmethod
    def _lazy_import_pyttsx3():
        try:
            import pyttsx3  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'pyttsx3' package is required for text-to-speech. "
                "Install it with `pip install pyttsx3`."
            ) from exc
        return pyttsx3

    @staticmethod
    def _lazy_import_sounddevice():
        try:
            import sounddevice  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'sounddevice' package is required for microphone access and playback. "
                "Install it with `pip install sounddevice`."
            ) from exc
        return sounddevice

    @staticmethod
    def _lazy_import_soundfile():
        try:
            import soundfile  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'soundfile' package is required to save and load audio files. "
                "Install it with `pip install soundfile`."
            ) from exc
        return soundfile
