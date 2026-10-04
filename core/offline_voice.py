"""Optional Vosk/SAPI voice hooks. Only explicit local assets; never downloads."""
from pathlib import Path
import queue
import threading


class OfflineVoice:
    def __init__(self, settings, respond, allowed, speaking, log, device=None):
        self.settings, self.respond, self.allowed = settings, respond, allowed
        self.speaking, self.log, self.device = speaking, log, device
        self.stop = threading.Event()
        self.interrupted = threading.Event()
        self.thread = None
        self.active = False

    def start(self):
        path = self.settings['vosk_model_path']
        if not path or not Path(path).is_dir():
            self.log('SYS: Offline microphone unavailable; configure a local Vosk model. Typed input is ready.')
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.interrupted.set()
        if self.thread:
            self.thread.join()  # caller awaits this off the event/UI threads

    def interrupt(self):
        self.interrupted.set()

    def _run(self):
        try:
            import sounddevice as sd
            from core.stt import VoskSTT
            stt = VoskSTT(model_path=self.settings['vosk_model_path'])
            speaker = None
            if self.settings['tts_enabled']:
                try:
                    import pythoncom
                    import win32com.client
                    pythoncom.CoInitialize()
                    speaker = win32com.client.Dispatch('SAPI.SpVoice')
                except Exception:
                    self.log('SYS: Local TTS unavailable; offline responses will be shown as text.')
            else:
                self.log('SYS: Local TTS disabled; offline responses will be shown as text.')
            chunks = queue.Queue(maxsize=32)
            def capture(data, frames, timing, status):
                if self.allowed() and not self.active:
                    try:
                        chunks.put_nowait(bytes(data))
                    except queue.Full:
                        pass
            with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16',
                                   blocksize=1600, device=self.device, callback=capture):
                self.log('SYS: Offline microphone ready (local Vosk).')
                while not self.stop.is_set():
                    try:
                        chunk = chunks.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if not self.allowed():
                        continue
                    text, final = stt.process_chunk(chunk)
                    if not final or not text:
                        continue
                    self.active = True
                    try:
                        self.log('You: ' + text)
                        answer = self.respond(text)
                        if speaker and answer and not self.stop.is_set():
                            self.interrupted.clear()
                            self.speaking(True)
                            try:
                                speaker.Speak(answer, 1)  # asynchronous native local voice
                                while not speaker.WaitUntilDone(50):
                                    if self.interrupted.is_set() or self.stop.is_set():
                                        speaker.Speak('', 3)  # async + purge
                                        break
                            finally:
                                self.speaking(False)
                    finally:
                        self.active = False
                        while not chunks.empty():
                            try:
                                chunks.get_nowait()
                            except queue.Empty:
                                break
        except Exception:
            self.log('SYS: Offline microphone unavailable; typed input remains ready.')
        finally:
            if 'pythoncom' in locals():
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
