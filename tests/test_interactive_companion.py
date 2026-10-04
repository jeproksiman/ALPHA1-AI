import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from core.context.companion import Companion
from core.offline_brain import OfflineBrain
from core import offline_voice, local_speech, stt
from memory.alpha_memory import AlphaMemory


@pytest.fixture
def store(tmp_path):
    memory=AlphaMemory(tmp_path/'memory.db')
    yield memory
    memory.close()


def test_whisper_first_and_no_download(monkeypatch, tmp_path):
    constructor=Mock(return_value=SimpleNamespace(transcribe=Mock()))
    monkeypatch.setattr(offline_voice,'local_whisper_model',lambda settings:str(tmp_path))
    monkeypatch.setattr(stt,'WhisperSTT',constructor)
    vosk=Mock();monkeypatch.setattr(stt,'VoskSTT',vosk)
    assert offline_voice.select_stt({'vosk_model_path':str(tmp_path)},Mock()) is constructor.return_value
    assert constructor.call_args.kwargs['allow_download'] is False
    vosk.assert_not_called()


def test_vosk_fallback_and_missing_status(monkeypatch,tmp_path):
    monkeypatch.setattr(offline_voice,'local_whisper_model',lambda settings:'cached')
    monkeypatch.setattr(stt,'WhisperSTT',Mock(side_effect=ImportError()))
    vosk=Mock();monkeypatch.setattr(stt,'VoskSTT',vosk)
    assert offline_voice.select_stt({'vosk_model_path':str(tmp_path)},Mock()) is vosk.return_value
    log=Mock()
    assert offline_voice.select_stt({},log) is None
    assert 'downloads are disabled' in log.call_args.args[0]


def test_whisper_model_cache_discovery_never_downloads(monkeypatch,tmp_path):
    monkeypatch.setenv('HF_HUB_CACHE',str(tmp_path))
    assert offline_voice.local_whisper_model({}) is None
    path=tmp_path/'models--Systran--faster-whisper-medium'/'snapshots'/'local'
    path.mkdir(parents=True)
    for file in ('model.bin','tokenizer.json','config.json'): (path/file).write_text('{}')
    assert offline_voice.local_whisper_model({})==str(path)
    assert offline_voice.local_whisper_model({'whisper_model':str(path)})==str(path)


def test_whisper_constructor_keeps_network_disabled(monkeypatch):
    import sys
    constructor=Mock()
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=constructor))
    model=stt.WhisperSTT('cached-model')
    assert constructor.call_args.kwargs['local_files_only'] is True
    assert constructor.call_args.kwargs['cpu_threads']==2
    assert constructor.call_args.kwargs['compute_type']=='int8'
    model.close()


def test_silence_segmentation_and_bounded_recordings():
    segmenter=offline_voice.UtteranceSegmenter(maximum=1,silence=.3)
    quiet=np.zeros(1600,dtype=np.int16).tobytes()
    voice=np.full(1600,1500,dtype=np.int16).tobytes()
    assert all(segmenter.feed(quiet) is None for _ in range(30))
    assert segmenter.feed(voice) is None
    segmenter.feed(quiet);segmenter.feed(quiet)
    utterance=segmenter.feed(quiet)
    assert isinstance(utterance,np.ndarray) and utterance.size<=16000
    emitted=[]
    for _ in range(100):
        item=segmenter.feed(voice)
        if item is not None: emitted.append(item)
    assert emitted and all(item.size<=16000 for item in emitted)


def test_piper_preferred_sapi_fallback(monkeypatch,tmp_path):
    path=tmp_path/'voice.onnx';path.write_bytes(b'placeholder')
    Path(str(path)+'.json').write_text('{}')
    piper,sapi=Mock(),Mock()
    monkeypatch.setattr(local_speech,'PiperSpeech',piper)
    monkeypatch.setattr(local_speech,'SapiSpeech',sapi)
    assert local_speech.select_speech({'piper_voice_path':str(path)},Mock()) is piper.return_value
    sapi.assert_not_called()
    piper.side_effect=ImportError()
    assert local_speech.select_speech({'piper_voice_path':str(path)},Mock()) is sapi.return_value
    assert local_speech.select_speech({'tts_enabled':False},Mock()) is None


def test_interrupt_aborts_pcm_and_toggles_same_speaking_hook(monkeypatch):
    import sys
    speaking,envelope=Mock(),Mock()
    output=local_speech.SpeechOutput({},speaking,envelope,Mock())
    engine=SimpleNamespace(chunks=lambda text,cancel:iter([(b'\x01\x00'*4000,16000)]))
    monkeypatch.setattr(local_speech,'select_speech',lambda settings,log:engine)
    stream=Mock()
    stream.__enter__=Mock(return_value=stream);stream.__exit__=Mock()
    stream.write.side_effect=lambda block:output.interrupt()
    monkeypatch.setitem(sys.modules,'sounddevice',SimpleNamespace(RawOutputStream=Mock(return_value=stream)))
    output.speak('A local answer')
    assert [call.args for call in speaking.call_args_list]==[(True,),(False,)]
    assert envelope.call_count==1
    stream.abort.assert_called_once()
    assert not output.busy
    assert output.lock.acquire(blocking=False)
    output.lock.release()


def test_no_overlapping_speech(monkeypatch):
    output=local_speech.SpeechOutput({},Mock(),Mock(),Mock())
    output.lock.acquire()
    select=Mock();monkeypatch.setattr(local_speech,'select_speech',select)
    output.speak('Queued response')
    select.assert_not_called()
    output.lock.release()


def test_live_topic_available_offline_and_survives_restart(store):
    companion=Companion(store,identity={})
    assert companion.record("We are improving ALPHA's offline voice today.",'Understood.','live')
    provider=Mock()
    brain=OfflineBrain(store,provider,settings={'semantic_enabled':False})
    assert 'offline voice' in brain.process('What were we just talking about?')['text']
    provider.generate.assert_not_called()
    another=AlphaMemory(store.path)
    try:
        restored=Companion(another,identity={})
        assert 'offline voice' in restored.handle('What were we just discussing?')
        assert another.recent_conversation()[0]['mode']=='live'
    finally: another.close()


def test_offline_goal_is_compact_live_and_ollama_context(store):
    provider=Mock()
    provider.generate.return_value={'ok':True,'text':'A concise local answer.'}
    brain=OfflineBrain(store,provider,settings={'semantic_enabled':False})
    brain.process('Remember that next I want to improve wake-word detection.')
    brain.process('Explain a missing Python module.')
    prompt=provider.generate.call_args.args[1]['summary']
    assert 'wake-word' in prompt and 'Next step:' in prompt
    assert len(prompt)<=1800
    assert 'wake-word' in brain.companion.prompt()  # same context is injected into both live providers
    assert store.recent_conversation()[0]['mode']=='offline'
    assert store.counts()['knowledge']==0  # conversation is not promoted to executable knowledge


def test_personal_facts_identity_and_forgetting(store):
    companion=Companion(store,identity={'assistant_name':'JARVIS','user_name':'Tester'})
    assert companion.handle('Call me Alex')=='I saved that locally.'
    companion.handle('I prefer concise answers')
    companion.handle('I am working on ALPHA')
    assert 'Alex' in companion.handle('what do you remember about me?')
    assert 'ALPHA' in companion.handle('what am I working on?')
    assert 'JARVIS' in companion.prompt()
    companion.record('Remember that we are testing offline voice.','Saved.','offline')
    companion.handle('Remember that we are testing offline voice.')
    companion.handle('forget that')
    assert not store.recent_conversation()
    assert not store.companion_facts().get('note')


def test_secrets_rejected_before_persistence_and_context(store):
    companion=Companion(store,identity={})
    secret='AIza'+'x'*35
    assert not companion.record('My API key is '+secret,'Okay.','live')
    assert not store.conversation('Hi',secret)
    assert not store.set_companion_fact('note',secret)
    assert not store.recent_conversation()
    assert secret not in companion.prompt()
    assert all(r['type']!='BLOB' for r in store.db.execute('PRAGMA table_info(interaction_history)'))


def test_cancelled_reasoning_never_records_stale_reply(store):
    cancel=threading.Event()
    provider=Mock()
    def generate(text,context):
        cancel.set()
        return {'ok':True,'text':'A stale result.'}
    provider.generate.side_effect=generate
    brain=OfflineBrain(store,provider,settings={'semantic_enabled':False})
    assert brain.process('An unresolved question',{'_cancel':cancel})['intent']=='cancelled'
    assert not store.recent_conversation()


def main_methods(names,ns):
    tree=ast.parse(Path('main.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JarvisLive')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'main.py','exec'),ns)
    return ns['JarvisLive']()


def test_live_callback_uses_same_store_and_face_bridge_exists(store):
    brain=OfflineBrain(store,Mock(),settings={})
    app=main_methods(['_remember_live_turn'],{})
    app._offline_brain=brain
    app._remember_live_turn('We are testing voice continuity.','Understood.')
    assert store.recent_conversation()[0]['mode']=='live'
    assert brain.context.recent().user_input=='We are testing voice continuity.'
    source=Path('main.py').read_text(encoding='utf-8')
    assert source.count('self._remember_live_turn(full_in,full_out)')==2
    assert 'self.ui.push_visemes(frames,hop,self._play_cursor)' in source
    assert 'parts.append(self._offline_brain.companion.prompt())' in source


def test_typed_and_voice_same_brain_and_persistence(store):
    provider=Mock();provider.generate.return_value={'ok':True,'text':'Local answer.'}
    brain=OfflineBrain(store,provider,settings={'semantic_enabled':False})
    app=main_methods(['_on_offline_text_command'],{'threading':threading})
    app._runtime=SimpleNamespace(offline=True)
    app._wake_enabled=False;app._offline_brain=brain;app.ui=Mock()
    app._local_lock=threading.Lock();app._local_busy=0;app._request_cancels=set()
    app._offline_voice=Mock()
    app._on_offline_text_command('Typed local request')
    app._on_offline_text_command('Voice local request',speak_reply=False)
    assert len(store.recent_conversation())==2
    app._offline_voice.speak.assert_called_once_with('Local answer.')


def test_cancelable_sequential_model_warmup():
    from core.providers.ollama_provider import OllamaProvider
    provider=OllamaProvider()
    event=threading.Event()
    provider.chat=Mock(side_effect=lambda *args,**kwargs:event.set())
    provider.embed=Mock()
    provider.warmup(event)
    provider.embed.assert_not_called()


def test_streaming_cancel_closes_local_model_response():
    from core.providers.ollama_provider import OllamaProvider
    provider=OllamaProvider(model='local')
    provider.status=lambda:{'available':True,'models':['local']}
    event=threading.Event()
    response=Mock(status_code=200)
    def chunks(**kwargs):
        event.set()
        yield json.dumps({'message':{'content':'stale'},'done':True}).encode()
    response.iter_lines.side_effect=chunks
    provider.session.post=Mock(return_value=response)
    assert not provider.chat([{'role':'user','content':'Hello'}],cancel=event)['ok']
    response.close.assert_called_once()


def test_pcm_envelope_drives_existing_face_and_waveform():
    import time
    pcm=np.full(800,5000,dtype=np.int16).tobytes()
    frames=[(.5,.4,0)]
    app=main_methods(['_offline_audio_envelope'],dict(np=np,time=time,
        _pcm_visemes=lambda samples,sr:frames,_VIS_HOP=480,_FIRST_SOUND=.03,
        _CURSOR_SLACK=.3,_pcm_level=lambda samples:.5))
    app._visemes=Mock();app._visemes.frames.return_value=frames
    app._play_cursor=0;app._out_latency=.2;app.ui=Mock();app._echo=Mock()
    app._offline_audio_envelope(pcm,16000)
    app.ui.push_visemes.assert_called_once()
    app.ui.set_audio_level.assert_called_once_with(.5)
    assert app._out_level==.5


def test_existing_profile_is_bounded_and_secret_filtered(store):
    companion=Companion(store,identity={})
    companion.profile_reader=lambda:{'preferences':{'format':{'value':'Concise replies'},
                                    'auth':{'value':'My password is unsafe'}}}
    assert 'Concise replies' in companion.prompt()
    assert 'unsafe' not in companion.prompt()


def test_live_personal_identity_and_goals_restore_offline(store):
    companion=Companion(store,identity={})
    companion.record('Call me Alex','Understood.','live')
    companion.record('Remember that next I want to improve wake-word detection.','Saved.','live')
    restored=Companion(store,identity={})
    assert restored.facts()['preferred_name']=='Alex'
    assert 'wake-word' in restored.facts()['goal']


def test_forget_does_not_delete_unrelated_goal(store):
    companion=Companion(store,identity={})
    companion.handle('Remember that next I want to improve wake-word detection.')
    companion.record('We are testing offline voice.','Understood.','offline')
    companion.handle('Forget that')
    assert 'wake-word' in companion.facts()['goal']


def test_verified_context_is_relevant_and_bounded(store):
    store.store_solution('Python missing module imports','Check the installed Python environment.',source='user',confidence=.9)
    provider=Mock();provider.generate.return_value={'ok':True,'text':'Local explanation.'}
    brain=OfflineBrain(store,provider,settings={'semantic_enabled':False})
    brain.process('Describe Python missing package issues from scratch')
    assert 'Relevant verified knowledge' in provider.generate.call_args.args[1]['summary']
    assert len(provider.generate.call_args.args[1]['summary'])<=2000
