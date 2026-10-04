"""
Speech-to-Text engines for MARK XL.

Whisper  – offline transcription via faster-whisper (VAD-buffered)
Vosk     – offline streaming transcription (lighter)
"""
import json
import numpy as np


class WhisperSTT:
    """Offline transcription using faster-whisper."""

    def __init__(self, model_name="small", language=None, device="auto", compute_type="auto",
                 allow_download=False, cpu_threads=2):
        from faster_whisper import WhisperModel
        # CPU/int8 keeps GPU memory available for Ollama; CUDA is explicit opt-in.
        device = "cpu" if device == "auto" else device
        compute_type = ("int8" if device == "cpu" else "float16") if compute_type == "auto" else compute_type
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                   local_files_only=not allow_download, cpu_threads=cpu_threads, num_workers=1)
        self._language = None if not language or language == "auto" else language

    def close(self):
        self._model = None

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,                       # greedy — 2-3x faster
                best_of=1,
                condition_on_previous_text=False,  # no hallucinations, faster
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            raise


class VoskSTT:
    """Streaming transcription using Vosk."""

    def __init__(self, model_path: str | None = None, language: str = "en-us"):
        from vosk import Model, KaldiRecognizer
        print("[STT] Loading Vosk model…")
        if model_path:
            model = Model(model_path)
        else:
            lang  = language.strip().lower() if language and language.strip().lower() != "auto" else "en-us"
            model = Model(lang=lang)
        self._rec = KaldiRecognizer(model, 16000)
        print("[STT] Vosk ready.")

    def process_chunk(self, audio_bytes: bytes) -> tuple[str, bool]:
        """Feed raw int16 LE PCM bytes. Returns (text, is_final)."""
        if self._rec.AcceptWaveform(audio_bytes):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False
