"""Memory-only synthetic voice smoke: no microphone recording or audio files."""
import sys
from pathlib import Path
import socket
import threading
from time import perf_counter
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.local_speech import SapiSpeech
from core.offline_voice import local_whisper_model
from core.stt import WhisperSTT
from memory.config_manager import get_runtime_settings


def main():
    cfg=get_runtime_settings()
    cfg['allow_download']=False
    path=local_whisper_model(cfg)
    if not path:
        print('No cached Whisper model. Configure ALPHA_WHISPER_MODEL with an installed model directory.')
        return 1
    model=engine=None
    original=socket.socket.connect
    try:
        socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Network disabled for local smoke'))
        engine=SapiSpeech()
        pcm,rate=next(engine.chunks('We are testing offline voice today.',threading.Event()))
        engine.close();engine=None
        audio=np.frombuffer(pcm,dtype=np.int16).astype(np.float32)/32768
        audio=np.interp(np.arange(int(len(audio)*16000/rate))*rate/16000,np.arange(len(audio)),audio).astype(np.float32)
        start=perf_counter()
        model=WhisperSTT(path,device=cfg['whisper_device'],compute_type=cfg['whisper_compute_type'])
        text=model.transcribe(audio)
        passed='offline' in text.lower() and 'voice' in text.lower()
        print('Local SAPI PCM synthesis: passed')
        print('Cached Whisper synthetic transcription:', 'passed' if passed else 'failed')
        print('Elapsed seconds:',round(perf_counter()-start,2))
        return 0 if passed else 1
    except Exception as error:
        print('Local voice smoke unavailable:',type(error).__name__)
        return 1
    finally:
        socket.socket.connect=original
        if model: model.close()
        if engine: engine.close()


if __name__=='__main__':
    raise SystemExit(main())
