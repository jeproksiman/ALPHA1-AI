import ast
import asyncio
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.runtime_mode import RuntimeMode, error_kind, provider_reachable
from memory import config_manager



@pytest.fixture(autouse=True)
def mock_heavy_stt(monkeypatch):
    from core import offline_voice
    def unavailable(settings,log):
        log('SYS: Offline microphone unavailable; Typed input is ready.')
        return None
    monkeypatch.setattr(offline_voice,'select_stt',unavailable)

def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize('mode,reachable,expected', [
    ('auto', True, True), ('auto', False, False),
    ('offline', True, False), ('live', False, True),
])
def test_cloud_admission(mode, reachable, expected):
    probe = AsyncMock(return_value=reachable)
    runtime = RuntimeMode(lambda: {'mode': mode}, probe=probe)
    assert run(runtime.ready('gemini', 'dummy')) is expected
    assert probe.await_count == (1 if mode == 'auto' else 0)


def test_quiet_recovery_waits_for_idle():
    now = [0]
    probe = AsyncMock(side_effect=[False, True])
    runtime = RuntimeMode(lambda: {'mode': 'auto'}, probe, lambda: now[0])
    assert not run(runtime.ready('gemini', 'dummy'))
    for second in range(1, 45):
        now[0] = second
        assert not run(runtime.ready('gemini', 'dummy'))
    assert probe.await_count == 1
    now[0] = 45
    assert not run(runtime.ready('gemini', 'dummy', idle=False))
    assert probe.await_count == 1
    assert run(runtime.ready('gemini', 'dummy'))
    assert not runtime.offline


def test_auth_is_distinct_and_waits_for_changed_key():
    probe = AsyncMock(return_value=True)
    runtime = RuntimeMode(lambda: {'mode': 'auto'}, probe)
    runtime.failed('auth', 'dummy')
    runtime.next_probe = 0
    assert not run(runtime.ready('gemini', 'dummy'))
    assert not run(runtime.ready('gemini', ''))
    probe.assert_not_awaited()
    assert run(runtime.ready('gemini', 'changed'))


@pytest.mark.parametrize('error,kind', [
    (socket.gaierror(11001, 'getaddrinfo failed'), 'network'),
    (TimeoutError(), 'network'), (ConnectionResetError(), 'network'),
    (ExceptionGroup('tasks', [socket.gaierror()]), 'network'),
    (RuntimeError('401 unauthorized'), 'auth'),
    (RuntimeError('API key not valid'), 'auth'),
    (ValueError('invalid model'), 'other'), (OSError('audio unavailable'), 'other'),
])
def test_failure_classification(error, kind):
    assert error_kind(error) == kind


def test_dns_probe_uses_actual_provider_and_no_auth(monkeypatch):
    connect = AsyncMock(side_effect=socket.gaierror())
    monkeypatch.setattr(asyncio, 'open_connection', connect)
    assert not run(provider_reachable('gemini'))
    assert connect.call_args.args == ('generativelanguage.googleapis.com', 443)
    assert not run(provider_reachable('deepgram'))
    assert connect.call_args.args == ('agent.deepgram.com', 443)


def test_probe_deadline_and_close(monkeypatch):
    async def blocked(*args, **kwargs):
        await asyncio.sleep(1)
    monkeypatch.setattr(asyncio, 'open_connection', blocked)
    assert not run(provider_reachable('gemini', timeout=.01))
    writer = Mock()
    monkeypatch.setattr(asyncio, 'open_connection', AsyncMock(return_value=(None, writer)))
    assert run(provider_reachable('gemini'))
    writer.close.assert_called_once()


def test_config_environment_and_private_config(monkeypatch):
    monkeypatch.delenv('ALPHA_RUNTIME_MODE', raising=False)
    monkeypatch.setattr(config_manager, 'load_api_keys', lambda: {})
    assert config_manager.get_runtime_settings()['mode'] == 'auto'
    monkeypatch.setattr(config_manager, 'load_api_keys', lambda: {'alpha_runtime_mode': 'offline'})
    assert config_manager.get_runtime_settings()['mode'] == 'offline'
    monkeypatch.setenv('ALPHA_RUNTIME_MODE', 'LIVE')
    assert config_manager.get_runtime_settings()['mode'] == 'live'
    monkeypatch.setenv('ALPHA_RUNTIME_MODE', 'invalid')
    assert config_manager.get_runtime_settings()['mode'] == 'auto'


def adapter(names, namespace=None):
    tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JarvisLive')
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    ns = {'asyncio': asyncio, 'get_output_device':lambda:'', **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'main.py', 'exec'), ns)
    return ns['JarvisLive']()


def test_offline_typed_and_explicit_live_prefix_use_local_brain():
    app = adapter(['_on_offline_text_command', '_on_text_command'])
    app._runtime = SimpleNamespace(offline=True)
    app._wake_enabled = False
    app.ui = Mock()
    app._local_busy = 0
    app._local_lock = threading.Lock()
    app._offline_brain = Mock()
    app._offline_brain.process.return_value = {'text': 'Local response'}
    app.session = Mock()
    assert app._on_offline_text_command('live: open calculator') == 'Local response'
    app._on_text_command('local status')
    assert app._offline_brain.process.call_args.args == ('local status',)
    app.session.send_client_content.assert_not_called()
    assert app._local_busy == 0


@pytest.mark.parametrize('mode,network,attempts', [('auto', True, 1), ('auto', False, 0),
                                                   ('offline', True, 0), ('live', False, 1)])
def test_real_run_loop_admission_without_cloud_or_ui_imports(monkeypatch, mode, network, attempts):
    from core.offline_voice import OfflineVoice
    settings = lambda: {'mode': mode, 'vosk_model_path': '', 'tts_enabled': False}
    client = Mock()
    class StopConnect:
        async def __aenter__(self):
            raise asyncio.CancelledError()
        async def __aexit__(self, *args):
            pass
    client.aio.live.connect.return_value = StopConnect()
    fake_dashboard = SimpleNamespace(DashboardServer=Mock(side_effect=RuntimeError('disabled in test')))
    monkeypatch.setitem(sys.modules, 'dashboard.server', fake_dashboard)
    ns = dict(RuntimeMode=RuntimeMode, get_runtime_settings=settings,
              _get_ai_provider=lambda: 'gemini', _get_api_key=lambda provider: 'dummy',
              OfflineVoice=OfflineVoice, audio_devices=Mock(), get_input_device=lambda: '',
              SEND_SAMPLE_RATE=16000, RECEIVE_SAMPLE_RATE=24000, confirm_gate=Mock(),
              set_trim_notifier=Mock(), _gemini=Mock(), genai=SimpleNamespace(Client=Mock(return_value=client)))
    app = adapter(['run', '_await_live_admission'], ns)
    app._runtime = RuntimeMode(settings, AsyncMock(return_value=network))
    app.ui = Mock(muted=False)
    app._local_lock = threading.Lock()
    app._local_busy = 0
    app._is_speaking = False
    app._ptt_enabled = False
    app._offline_voice = None
    app.set_speaking = Mock()
    app._on_offline_text_command = Mock()
    app._active_provider = 'gemini'
    app._resume_handle = None
    app._enhanced_live = False
    app._build_config = Mock()
    app._session_log = []
    async def exercise():
        task = asyncio.create_task(app.run())
        await asyncio.sleep(.04)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(exercise())
    assert client.aio.live.connect.call_count == attempts
    if not attempts:
        assert app.ui.write_log.call_count == 2  # one mode message, one honest mic status
        app.ui.set_state.assert_called_with('OFFLINE')


def test_ui_existing_face_state_and_backend_startup_gate():
    source = Path('ui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_apply_state')
    assert ast.unparse(method) == "def _apply_state(self, state: str):\n    self.hud.state = state\n    self.hud.speaking = state == 'SPEAKING'"
    assert "if not self._ready and get_runtime_settings()['mode'] == 'live':" in source
    assert 'a VPN may be required' not in Path('main.py').read_text(encoding='utf-8')


def test_active_live_network_loss_closes_session_and_enters_offline(monkeypatch):
    from core.offline_voice import OfflineVoice
    settings = lambda: {'mode': 'auto', 'vosk_model_path': '', 'tts_enabled': False}
    closed = Mock()
    class Session:
        async def __aenter__(self):
            return Mock()
        async def __aexit__(self, *args):
            closed()
    client = Mock()
    client.aio.live.connect.return_value = Session()
    monkeypatch.setitem(sys.modules, 'dashboard.server', SimpleNamespace(
        DashboardServer=Mock(side_effect=RuntimeError('disabled in test'))))
    ns = dict(get_runtime_settings=settings, OfflineVoice=OfflineVoice,
        _get_ai_provider=lambda: 'gemini', _get_api_key=lambda provider: 'dummy',
        audio_devices=Mock(), get_input_device=lambda: '', SEND_SAMPLE_RATE=16000,
        RECEIVE_SAMPLE_RATE=24000, confirm_gate=Mock(), set_trim_notifier=Mock(),
        _gemini=Mock(), genai=SimpleNamespace(Client=Mock(return_value=client)),
        error_kind=error_kind, _is_reconnect_signal=lambda error: False)
    app = adapter(['run', '_await_live_admission'], ns)
    app._runtime = RuntimeMode(settings, AsyncMock(return_value=True))
    app.ui = Mock(muted=False)
    app._local_lock, app._local_busy = threading.Lock(), 0
    app._is_speaking = app._ptt_enabled = app._wake_enabled = False
    app._offline_voice = None
    app._active_provider = 'gemini'
    app._resume_handle = None
    app._enhanced_live = False
    app._briefing_sent = True
    app._session_log = []
    app.set_speaking = app._on_offline_text_command = app._build_config = Mock()
    async def wait():
        await asyncio.Event().wait()
    async def lose_network():
        raise socket.gaierror(11001, 'getaddrinfo failed')
    for name in ('_watch_reconnect', '_watch_runtime_mode', '_send_realtime', '_listen_audio',
                 '_play_audio', '_run_system_monitor', '_run_background_monitor',
                 '_run_proactive_mode', '_run_sleep_watch'):
        setattr(app, name, wait)
    app._receive_audio = lose_network
    async def exercise():
        task = asyncio.create_task(app.run())
        await asyncio.sleep(.04)
        assert app._runtime.offline and app.session is None
        assert client.aio.live.connect.call_count == 1
        assert app._runtime.probe.await_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(exercise())
    closed.assert_called_once()
    assert not any('retrying' in str(call).lower() for call in app.ui.write_log.call_args_list)


def test_voice_requires_local_assets_and_never_downloads(monkeypatch):
    from core.offline_voice import OfflineVoice
    log = Mock()
    voice = OfflineVoice({'vosk_model_path': '', 'tts_enabled': True}, Mock(), Mock(), Mock(), log)
    from core import offline_voice
    voice._run()
    assert not voice.ready
    assert any('Typed input is ready' in str(call) for call in log.call_args_list)



def test_local_memory_commands_reasoning_and_model_failure(tmp_path):
    from core.offline_brain import OfflineBrain
    from memory.alpha_memory import AlphaMemory
    memory = AlphaMemory(tmp_path / 'test.db')
    provider = Mock()
    provider.generate.return_value = {'ok': True, 'text': 'Local reasoning answer.'}
    try:
        brain = OfflineBrain(memory, provider, settings={})
        action = Mock(return_value='Calculator opened locally.')
        brain.register_command('open calculator', action)
        memory.store_solution('what is my local preference', 'Local storage.', source='user', confidence=.95)
        assert brain.process('open calculator')['text'] == 'Calculator opened locally.'
        assert brain.process('what is my local preference')['source'] == 'memory'
        provider.generate.assert_not_called()
        assert brain.process('explain ModuleNotFoundError')['text'] == 'Local reasoning answer.'
        provider.generate.return_value = {'ok': False, 'text': 'Local Ollama unavailable.'}
        assert brain.process('open calculator')['source'] == 'local'
        assert brain.process('what is my local preference')['source'] == 'memory'
        assert brain.process('different unknown reasoning')['text']
    finally:
        memory.close()


def test_local_voice_routes_transcript_and_reuses_speaking_state(monkeypatch, tmp_path):
    from core import offline_voice
    from core.offline_voice import OfflineVoice
    respond = Mock(return_value='Local spoken response')
    voice = OfflineVoice({'vosk_model_path':str(tmp_path),'tts_enabled':True},respond,
                         lambda:True,Mock(),Mock())
    recognizer=Mock()
    recognizer.process_chunk.return_value=('open calculator',True)
    monkeypatch.setattr(offline_voice,'select_stt',lambda settings,log:recognizer)
    class Stream:
        def __init__(self,**kwargs): self.callback=kwargs['callback']
        def __enter__(self):
            self.callback(b'\x01\x00'*1600,1600,None,None)
            return self
        def __exit__(self,*args): pass
    monkeypatch.setitem(sys.modules,'sounddevice',SimpleNamespace(RawInputStream=Stream))
    def spoken(text): voice.stop.set()
    voice.output.speak=Mock(side_effect=spoken)
    voice._run()
    respond.assert_called_once_with('open calculator')
    voice.output.speak.assert_called_once_with('Local spoken response')


def test_recovery_rechecks_turn_started_during_probe():
    from core.offline_voice import OfflineVoice
    settings = lambda: {'mode': 'auto', 'vosk_model_path': '', 'tts_enabled': False}
    app = adapter(['_await_live_admission'], dict(get_runtime_settings=settings,
        _get_ai_provider=lambda: 'gemini', _get_api_key=lambda provider: 'dummy',
        OfflineVoice=OfflineVoice, audio_devices=Mock(), get_input_device=lambda: ''))
    now, calls = [0], [0]
    async def probe(provider):
        calls[0] += 1
        if calls[0] == 1:
            return False
        if calls[0] == 2:
            app._local_busy = 1  # request starts while the probe is in flight
        return True
    app._runtime = RuntimeMode(settings, probe, lambda: now[0])
    app._local_lock, app._local_busy = threading.Lock(), 0
    app._is_speaking, app._offline_voice = False, None
    app.ui = Mock()
    app.set_speaking = app._on_offline_text_command = Mock()
    async def exercise():
        task = asyncio.create_task(app._await_live_admission())
        await asyncio.sleep(.02)
        now[0] = 45
        await asyncio.sleep(1.05)
        assert app._runtime.offline and not task.done()
        app._local_busy = 0
        await asyncio.wait_for(task, 2)
        assert not app._runtime.offline
    run(exercise())
