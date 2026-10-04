"""Local TTS selection and interruptible PCM playback; no audio files are saved."""
from pathlib import Path
import threading
import numpy as np


class PiperSpeech:
    def __init__(self, path):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(path), use_cuda=False)

    def chunks(self, text, cancel):
        for chunk in self.voice.synthesize(text):
            if cancel.is_set():
                return
            yield chunk.audio_int16_bytes, chunk.sample_rate


class SapiSpeech:
    def __init__(self):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        self.com = pythoncom
        self.speaker = win32com.client.Dispatch('SAPI.SpVoice')
        self.memory = win32com.client.Dispatch('SAPI.SpMemoryStream')
        self.memory.Format.Type = 22  # SAFT22kHz16BitMono
        self.speaker.AudioOutputStream = self.memory

    def chunks(self, text, cancel):
        self.memory.Seek(0, 0)
        self.speaker.Speak(text, 17)  # asynchronous + literal text, never SSML
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


def select_speech(settings, log):
    if not settings.get('tts_enabled', True):
        return None
    provider = settings.get('tts_provider', 'auto')
    path = Path(settings.get('piper_voice_path') or '__no_piper_voice__')
    if provider in ('auto', 'piper') and path.is_file() and Path(str(path)+'.json').is_file():
        try:
            return PiperSpeech(path)
        except Exception:
            log('SYS: Piper unavailable; trying local Windows speech.')
    try:
        return SapiSpeech()
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
