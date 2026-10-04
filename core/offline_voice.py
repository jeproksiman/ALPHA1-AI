"""Bounded local listening and speech, with one companion callback for text/voice."""
from pathlib import Path
import os
import queue
import threading
import time
import numpy as np
from core.local_speech import SpeechOutput
from core.learning.knowledge_extractor import contains_secret


def local_whisper_model(settings):
    model = str(settings.get('whisper_model','small'))
    def complete(path):
        return all((path/name).is_file() for name in ('model.bin','config.json','tokenizer.json'))
    if complete(Path(model)):
        return model
    root = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub')))
    preferred = list((root/('models--Systran--faster-whisper-'+model)/'snapshots').glob('*'))
    for path in preferred:
        if complete(path):
            return str(path)
    # Reuse the smallest already installed model instead of downloading the default.
    cached = [p for p in root.glob('models--Systran--faster-whisper-*/snapshots/*') if complete(p)]
    if cached:
        return str(min(cached,key=lambda p:(p/'model.bin').stat().st_size))
    return model if settings.get('allow_download',False) else None


def select_stt(settings, log):
    from core.stt import WhisperSTT, VoskSTT
    if settings.get('stt_provider','faster-whisper') in ('auto','faster-whisper'):
        try:
            model = local_whisper_model(settings)
            if model:
                stt = WhisperSTT(model, device=settings.get('whisper_device','auto'),
                                 compute_type=settings.get('whisper_compute_type','auto'),
                                 allow_download=settings.get('allow_download',False))
                log('SYS: Local STT ready (faster-whisper, cached model).')
                return stt
        except Exception:
            pass
    path = settings.get('vosk_model_path','')
    if path and Path(path).is_dir():
        try:
            stt = VoskSTT(model_path=path)
            log('SYS: Local STT ready (Vosk).')
            return stt
        except Exception:
            pass
    log('SYS: Offline microphone unavailable. Install faster-whisper and configure a local '
        'ALPHA_WHISPER_MODEL directory, or a Vosk model. Typed input is ready; downloads are disabled by default.')
    return None


class UtteranceSegmenter:
    def __init__(self, rate=16000, threshold=.012, silence=.65, maximum=12):
        self.rate, self.threshold, self.silence, self.maximum = rate, threshold, silence, maximum
        self.reset()

    def reset(self):
        self.parts, self.samples, self.quiet = [], 0, 0
        self.pre = None

    def feed(self, pcm):
        audio = np.frombuffer(pcm,dtype=np.int16).astype(np.float32)/32768
        level = float(np.sqrt(np.mean(audio*audio))) if audio.size else 0
        if not self.parts and level < self.threshold:
            self.pre = pcm
            return None
        if not self.parts and self.pre:
            self.parts.append(self.pre)
            self.samples += len(self.pre)//2
        self.parts.append(pcm)
        self.samples += len(pcm)//2
        self.quiet = self.quiet + len(pcm)//2 if level < self.threshold else 0
        if self.quiet/self.rate >= self.silence or self.samples/self.rate >= self.maximum:
            utterance = np.frombuffer(b''.join(self.parts),dtype=np.int16).astype(np.float32)/32768
            self.reset()
            return utterance
        return None


class OfflineVoice:
    def __init__(self, settings, respond, allowed, speaking, log, device=None,
                 envelope=None, output_device=None, cancel_response=None, speech_text=None, warmup=None,
                 audio_observer=None):
        self.settings, self.respond, self.allowed = settings, respond, allowed
        self.speaking, self.log, self.device = speaking, log, device
        self.stop, self.interrupted = threading.Event(), threading.Event()
        self.thread = None
        self.active = False
        self.ready = False
        self.speech_text = speech_text
        self.warmup = warmup
        self.warm_cancel = threading.Event()
        self.cancel_response = cancel_response
        self.audio_observer = audio_observer
        self.remote_until = 0.0
        self.output = SpeechOutput(settings,speaking,envelope or (lambda pcm,rate:None),log,output_device)
        self.chunks = queue.Queue(maxsize=32)

    def start(self):
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def submit_remote(self, pcm):
        """Authenticated dashboard 16 kHz mono PCM, using the same local STT."""
        if not isinstance(pcm, bytes) or not 0 < len(pcm) <= 32000 or len(pcm) % 2:
            return
        self.remote_until = time.monotonic() + .5
        if self.ready and self.allowed() and not self.active and not self.output.busy:
            try:
                self.chunks.put_nowait(pcm)
            except queue.Full:
                pass

    def close(self):
        self.stop.set()
        self.interrupt()
        if self.thread:
            self.thread.join()

    def interrupt(self):
        self.interrupted.set()
        self.warm_cancel.set()
        self.output.interrupt()
        if self.cancel_response:
            self.cancel_response()
        self._drain()

    def _drain(self):
        while True:
            try: self.chunks.get_nowait()
            except queue.Empty: break

    def speak(self,text):
        if not self.stop.is_set() and not self.interrupted.is_set():
            if self.speech_text:
                self.speech_text(text)
            self.output.speak(text)

    def _run(self):
        stt = None
        try:
            import sounddevice as sd
            stt = select_stt(self.settings,self.log)
            from core.local_speech import select_speech
            speech = select_speech(self.settings,self.log)
            if speech:
                self.log('SYS: [TTS] Local voice ready.')
                if hasattr(speech,'close'):
                    speech.close()
            if stt is None or self.stop.is_set():
                return
            if self.settings.get('warmup',True) and hasattr(stt,'transcribe') and not self.stop.is_set():
                stt.transcribe(np.zeros(1600,dtype=np.float32))
            segmenter = UtteranceSegmenter()
            last_active = time.monotonic()
            warmed = False
            def capture(data,frames,timing,status):
                if time.monotonic() < self.remote_until:
                    return
                if self.audio_observer:
                    self.audio_observer(bytes(data))
                # Barge-in is opt-in because speaker echo varies by room.
                if self.output.busy and self.settings.get('barge_in',False):
                    pcm = np.frombuffer(bytes(data),dtype=np.int16).astype(np.float32)/32768
                    if pcm.size and float(np.sqrt(np.mean(pcm*pcm))) > .12:
                        self.interrupt()
                if self.allowed() and not self.active and not self.output.busy:
                    audio = np.frombuffer(bytes(data),dtype=np.int16).astype(np.float32)/32768
                    if audio.size and float(np.sqrt(np.mean(audio*audio)))>=.012:
                        self.warm_cancel.set()
                    try: self.chunks.put_nowait(bytes(data))
                    except queue.Full: pass
            with sd.RawInputStream(samplerate=16000,channels=1,dtype='int16',blocksize=1600,
                                   device=self.device,callback=capture):
                self.ready=True
                self.log('SYS: Offline microphone ready.')
                while not self.stop.is_set():
                    if stt is not None and hasattr(stt,'close') and time.monotonic()-last_active>300:
                        stt.close();stt=None
                    try: chunk = self.chunks.get(timeout=.2)
                    except queue.Empty:
                        if self.settings.get('warmup',True) and self.warmup and not warmed and time.monotonic()-last_active>5:
                            warmed=True
                            self.warm_cancel.clear()
                            threading.Thread(target=self.warmup,args=(self.warm_cancel,),daemon=True).start()
                        # Release Whisper after extended idle; reload only when speech arrives.
                        if stt is not None and hasattr(stt,'close') and time.monotonic()-last_active>300:
                            stt.close();stt=None
                        continue
                    if not self.allowed():
                        segmenter.reset()
                        continue
                    if stt is None:
                        audio = np.frombuffer(chunk,dtype=np.int16).astype(np.float32)/32768
                        if not audio.size or float(np.sqrt(np.mean(audio*audio)))<.012:
                            continue
                        stt=select_stt(self.settings,self.log)
                        if stt is None: return
                        last_active=time.monotonic()
                    if hasattr(stt,'process_chunk'):
                        self.interrupted.clear()
                        text, final = stt.process_chunk(chunk)
                        if not final: continue
                    else:
                        audio=segmenter.feed(chunk)
                        if audio is None: continue
                        self.active=True
                        self.interrupted.clear()
                        text=stt.transcribe(audio)
                    if not text or self.stop.is_set() or self.interrupted.is_set():
                        self.active=False
                        continue
                    self.active=True
                    self.interrupted.clear()
                    try:
                        self.log('You: '+('Sensitive input withheld.' if contains_secret(text) else text))
                        answer=self.respond(text)
                        if answer and not self.interrupted.is_set():
                            self.speak(answer)
                    finally:
                        self.active=False
                        last_active=time.monotonic()
                        segmenter.reset()
                        self._drain()
        except Exception:
            self.log('SYS: Offline microphone could not start; typed input remains ready.')
        finally:
            self.ready=False
            if stt is not None and hasattr(stt,'close'):
                stt.close()
