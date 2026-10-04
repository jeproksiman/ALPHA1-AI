"""Local TTS selection and interruptible PCM playback; no audio files are saved."""
from pathlib import Path
import threading
import math
from xml.sax.saxutils import escape
import numpy as np


def setting_number(settings, name, default, low, high):
    try:
        value = float(settings.get(name, default))
        return max(low, min(high, value)) if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


class PiperSpeech:
    def __init__(self, path, settings=None):
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        settings = settings or {}
        self.voice = PiperVoice.load(str(path), use_cuda=False)
        speaker = str(settings.get('tts_voice') or '')
        speaker_id = int(speaker) if speaker.isdigit() else self.voice.config.speaker_id_map.get(speaker)
        self.config = SynthesisConfig(speaker_id=speaker_id,
            length_scale=1 / setting_number(settings, 'tts_rate', 1, .5, 2))

    def chunks(self, text, cancel):
        for chunk in self.voice.synthesize(text, syn_config=self.config):
            if cancel.is_set():
                return
            yield chunk.audio_int16_bytes, chunk.sample_rate


class SapiSpeech:
    def __init__(self, settings=None):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        self.com = pythoncom
        self.speaker = self.memory = None
        try:
            self._configure(settings)
        except Exception:
            self.close()
            raise

    def _configure(self, settings):
        import win32com.client
        self.speaker = win32com.client.Dispatch('SAPI.SpVoice')
        settings = settings or {}
        self.speaker.Rate = max(-10, min(10, round(5 * math.log2(setting_number(settings, 'tts_rate', 1, .5, 2)))))
        self.pitch = round(setting_number(settings, 'tts_pitch', 0, -10, 10))
        selected = str(settings.get('tts_voice') or '').casefold()
        if selected:
            for voice in self.speaker.GetVoices():
                if selected in voice.GetDescription().casefold() or selected == voice.Id.casefold():
                    self.speaker.Voice = voice
                    break
            else:
                raise ValueError('Requested SAPI voice is not installed')
        self.memory = win32com.client.Dispatch('SAPI.SpMemoryStream')
        self.memory.Format.Type = 22  # SAFT22kHz16BitMono
        self.speaker.AudioOutputStream = self.memory

    def chunks(self, text, cancel):
        self.memory.Seek(0, 0)
        # Only our escaped text enters SAPI markup; input cannot inject tags.
        if self.pitch:
            self.speaker.Speak(f'<pitch absmiddle="{self.pitch}">{escape(text)}</pitch>', 9)
        else:
            self.speaker.Speak(text, 17)  # asynchronous + literal text
        while not self.speaker.WaitUntilDone(30):
            if cancel.is_set():
                self.speaker.Speak('', 3)
                return
        if not cancel.is_set():
            data = bytes(self.memory.GetData())
            # SpMemoryStream can retain a previous longer utterance; use current position.
            length = int(self.memory.Seek(0, 1))
            yield data[:length], 22050

    def close(self):
        self.speaker = self.memory = None
        self.com.CoUninitialize()


class KokoroSpeech:
    """Kokoro ONNX from explicit local files; no download or package upgrade."""
    def __init__(self, settings):
        from kokoro_onnx import Kokoro
        self.engine = Kokoro(settings['kokoro_model_path'], settings['kokoro_voices_path'])
        self.voice = settings.get('tts_voice') or 'af_heart'
        self.rate = setting_number(settings, 'tts_rate', 1, .5, 2)

    def chunks(self, text, cancel):
        # Sentence chunks provide cancellation points and prompt first audio.
        import re
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            if cancel.is_set():
                return
            samples, rate = self.engine.create(sentence, voice=self.voice, speed=self.rate,
                                               lang='en-gb' if self.voice.startswith('b') else 'en-us')
            if not cancel.is_set():
                yield (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes(), rate


def select_speech(settings, log):
    if not settings.get('tts_enabled', True):
        return None
    provider = settings.get('tts_provider', 'auto')
    path = Path(settings.get('piper_voice_path') or '__no_piper_voice__')
    if provider in ('auto', 'piper') and path.is_file() and Path(str(path)+'.json').is_file():
        try:
            return PiperSpeech(path, settings)
        except Exception:
            log('SYS: Piper unavailable; trying local Windows speech.')
    if provider in ('auto', 'kokoro') and all(Path(settings.get(key) or '__missing__').is_file()
            for key in ('kokoro_model_path', 'kokoro_voices_path')):
        try:
            return KokoroSpeech(settings)
        except Exception:
            log('SYS: Local Kokoro unavailable; trying Windows speech.')
    if provider not in ('auto', 'piper', 'kokoro', 'sapi'):
        log('SYS: Unknown local TTS provider; typed responses remain ready.')
        return None
    try:
        return SapiSpeech(settings)
    except Exception:
        log('SYS: Local speech unavailable; typed responses remain ready.')
        return None


class SpeechOutput:
    def __init__(self, settings, speaking, envelope, log, device=None):
        self.settings, self.speaking, self.envelope = settings, speaking, envelope
        self.log, self.device = log, device
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.busy = False

    def interrupt(self):
        self.cancel.set()

    def speak(self, text):
        # One output owner, with no backlog of stale answers.
        if not self.lock.acquire(blocking=False):
            return
        self.cancel.clear()
        self.busy = True
        engine = None
        try:
            import sounddevice as sd
            engine = select_speech(self.settings, self.log)
            if engine is None:
                return
            self.speaking(True)
            for pcm, rate in engine.chunks(text[:2000], self.cancel):
                if self.cancel.is_set():
                    break
                volume = setting_number(self.settings, 'tts_volume', 1, 0, 1)
                if volume != 1:
                    pcm = (np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * volume).astype(np.int16).tobytes()
                # Match the existing live output/device-picker sample rate.
                if rate != 24000:
                    samples=np.frombuffer(pcm,dtype=np.int16)
                    if not samples.size:
                        continue
                    pcm=np.interp(np.arange(int(samples.size*24000/rate))*rate/24000,
                                  np.arange(samples.size),samples).astype(np.int16).tobytes()
                    rate=24000
                with sd.RawOutputStream(samplerate=rate, channels=1, dtype='int16',
                                        blocksize=max(1024,int(rate*.04)), device=self.device) as stream:
                    # The existing live mouth analyser uses a 1024-sample window.
                    step = max(1024,int(rate*.04))*2
                    for start in range(0,len(pcm),step):
                        if self.cancel.is_set():
                            stream.abort()
                            break
                        block = pcm[start:start+step]
                        self.envelope(block,rate)
                        stream.write(block)
        except Exception:
            self.log('SYS: Local speech could not complete; the response is available as text.')
        finally:
            try:
                self.speaking(False)
                if engine and hasattr(engine,'close'):
                    engine.close()
            finally:
                self.busy = False
                self.lock.release()
