import ast
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.action_loader import ActionRegistry, ActionRecord
from core.brain.action_planner import ActionPlanner, validate_args
from core.offline_brain import OfflineBrain
from memory.alpha_memory import AlphaMemory
from memory import config_manager


@pytest.fixture
def setup(tmp_path):
    handler = Mock(return_value='Found a verified current headline: Example announcement.')
    records = {
        'web_search': ActionRecord('web_search', 'Search the web',
            {'type': 'OBJECT', 'properties': {'query': {'type': 'STRING'}}, 'required': ['query']}, handler, valid=True),
        'file_controller': ActionRecord('file_controller', 'Manage files',
            {'type': 'OBJECT', 'properties': {'action': {'type': 'STRING', 'enum': ['list', 'delete']}},
             'required': ['action']}, handler, valid=True),
    }
    registry = ActionRegistry(records, Mock())
    provider = Mock()
    provider.generate.return_value = {'ok': True, 'text': 'A local conversation answer.'}
    provider.chat.side_effect = [
        {'ok': True, 'text': json.dumps({'type': 'action', 'action': 'web_search', 'args': {'query': 'example'}})},
        {'ok': True, 'text': 'The retrieved headline is Example announcement.'}]
    memory = AlphaMemory(tmp_path / 'memory.db')
    brain = OfflineBrain(memory, provider, settings={'semantic_enabled': False})
    brain.action_planner = ActionPlanner(provider, registry)
    yield brain, provider, registry, handler
    memory.close()


def test_chat_goes_directly_to_primary(setup):
    brain, provider, _, handler = setup
    assert brain.process('Hello Jarvis')['text'] == 'A local conversation answer.'
    provider.generate.assert_called_once()
    provider.chat.assert_not_called()
    handler.assert_not_called()


def test_fast_selection_existing_registry_and_primary_summary(setup):
    brain, provider, registry, handler = setup
    result = brain.process('Search for example')
    assert result['intent'] == 'action_result'
    assert provider.chat.call_args_list[0].kwargs['role'] == 'fast'
    assert provider.chat.call_args_list[1].kwargs['role'] == 'primary'
    handler.assert_called_once()
    assert handler.call_args.kwargs['parameters'] == {'query': 'example'}
    assert 'verified current headline' in str(provider.chat.call_args_list[1])
    assert brain.memory.recent_conversation()[0]['response'] == result['text']
    assert len(brain.memory.recent_conversation()) == 1


def test_complex_selection_uses_primary(setup):
    brain, provider, _, _ = setup
    provider.chat.side_effect = [
        {'ok': True, 'text': '{"type":"action","action":"web_search","args":{"query":"example"}}'},
        {'ok': True, 'text': '{"type":"answer","text":"Here are the retrieved implications."}'}]
    assert brain.process('Research the announcement and then explain its implications')['text'] == 'Here are the retrieved implications.'
    assert provider.chat.call_args_list[0].kwargs['role'] == 'primary'
    assert provider.chat.call_args_list[1].kwargs['role'] == 'primary'


@pytest.mark.parametrize('plan', [
    {'type': 'action', 'action': 'unregistered_shell', 'args': {}},
    {'type': 'action', 'action': 'web_search', 'args': {'query': 2}},
    {'type': 'action', 'action': 'web_search', 'args': {'query': 'ok', 'command': 'cmd'}},
    {'type': 'action', 'action': 'web_search', 'args': {}},
    {'type': 'action', 'action': 'file_controller', 'args': {'action': 'format'}},
])
def test_invalid_plans_cannot_execute(setup, plan):
    brain, provider, _, handler = setup
    provider.chat.side_effect = [{'ok': True, 'text': json.dumps(plan)}]
    assert brain.process('Search for example')['intent'] == 'invalid_action'
    handler.assert_not_called()


def test_dangerous_action_parks_until_confirmation(setup):
    brain, provider, _, handler = setup
    provider.chat.side_effect = [
        {'ok': True, 'text': '{"type":"action","action":"file_controller","args":{"action":"delete"}}'},
        {'ok': True, 'text': 'Please confirm on the HUD.'},
        {'ok': True, 'text': 'The action result has been reported.'}]
    confirmer = Mock(return_value='[CONFIRMATION_PENDING] Confirm the action before execution.')
    brain.action_planner.confirmer = confirmer
    assert brain.process('Delete a file')['intent'] == 'confirmation_required'
    handler.assert_not_called()
    confirmer.call_args.args[2]()
    handler.assert_called_once()
    assert handler.call_args.kwargs['alpha_confirmed'] is True


def test_web_disabled_does_not_disable_local_action(setup):
    brain, provider, _, handler = setup
    brain.action_planner.settings = lambda: {'web_tools_enabled': False}
    assert brain.process('Search for example')['intent'] == 'web_disabled'
    handler.assert_not_called()
    brain.register_command('local ping', lambda ctx: 'pong')
    provider.chat.reset_mock()
    assert brain.process('local ping')['text'] == 'pong'
    provider.chat.assert_not_called()


def test_fresh_question_cannot_use_model_recall(setup):
    brain, provider, _, handler = setup
    provider.chat.side_effect = [
        {'ok': True, 'text': '{"type":"answer","text":"An unverified current headline"}'},
        {'ok': True, 'text': 'Retrieved current headline.'}]
    brain.process('What is the latest news today?')
    handler.assert_called_once()
    assert handler.call_args.kwargs['parameters']['query'] == 'What is the latest news today?'


def test_cancelled_plan_never_executes(setup):
    brain, provider, _, handler = setup
    event = threading.Event()
    def cancelled(*args, **kwargs):
        event.set()
        return {'ok': True, 'text': '{"type":"action","action":"web_search","args":{"query":"example"}}'}
    provider.chat.side_effect = cancelled
    assert brain.process('Search example', {'_cancel': event})['intent'] == 'cancelled'
    handler.assert_not_called()


def test_no_keys_required_and_new_voice_settings(monkeypatch):
    monkeypatch.setattr(config_manager, 'load_api_keys', lambda: {})
    for name in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'DEEPGRAM_API_KEY', 'ANTHROPIC_API_KEY', 'ALPHA_CLOUD_AI_ENABLED'):
        monkeypatch.delenv(name, raising=False)
    cfg = config_manager.get_runtime_settings()
    assert cfg['brain_provider'] == 'ollama' and cfg['cloud_ai_enabled'] is False
    monkeypatch.setenv('ALPHA_TTS_PROVIDER', 'sapi')
    monkeypatch.setenv('ALPHA_STT_PROVIDER', 'vosk')
    monkeypatch.setenv('ALPHA_TTS_VOICE', 'David')
    monkeypatch.setenv('ALPHA_TTS_RATE', '1.2')
    cfg = config_manager.get_runtime_settings()
    assert (cfg['tts_provider'], cfg['stt_provider'], cfg['tts_voice'], cfg['tts_rate']) == ('sapi', 'vosk', 'David', '1.2')


def test_legacy_generation_helpers_use_only_ollama(monkeypatch):
    from core import gemini
    from core.providers import ollama_provider
    monkeypatch.setattr(config_manager, 'load_api_keys', lambda: {})
    model = Mock()
    model.chat.return_value = {'ok': True, 'text': 'Local helper output.'}
    monkeypatch.setattr(ollama_provider, 'OllamaProvider', Mock(return_value=model))
    monkeypatch.setattr(gemini, 'api_key', Mock(side_effect=AssertionError('Cloud key accessed')))
    assert gemini.call('Explain this code', tier=gemini.SMART).text == 'Local helper output.'
    assert model.chat.call_args.kwargs['role'] == 'primary'
    assert gemini.call('Current news', tier=gemini.SEARCH) is None


def test_no_cloud_sdk_imports_in_normal_main():
    tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.ImportFrom) and str(node.module).startswith('google') for node in tree.body)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JarvisLive')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run')
    assert '_run_local' in ast.unparse(method)
    assert 'live.connect' not in ast.unparse(method)


def test_pitch_volume_and_invalid_settings():
    from core.local_speech import setting_number
    assert setting_number({'tts_volume': 'nan'}, 'tts_volume', 1, 0, 1) == 1
    assert setting_number({'tts_rate': '100'}, 'tts_rate', 1, .5, 2) == 2
    assert setting_number({'tts_pitch': '-2'}, 'tts_pitch', 0, -10, 10) == -2


def test_existing_web_search_skips_ai_backends(monkeypatch):
    from actions import web_search
    monkeypatch.setattr(config_manager, 'load_api_keys', lambda: {})
    monkeypatch.delenv('ALPHA_CLOUD_AI_ENABLED', raising=False)
    retrieve = Mock(return_value=[{'title': 'Retrieved', 'snippet': 'Verified tool result', 'url': 'https://example.com'}])
    monkeypatch.setattr(web_search, '_ddg_search', retrieve)
    monkeypatch.setattr(web_search, '_gemini_search', Mock(side_effect=AssertionError('Cloud search attempted')))
    assert 'Retrieved' in web_search._search('synthetic query')
    retrieve.assert_called_once_with('synthetic query')


def test_generated_shell_and_file_escape_are_rejected(tmp_path):
    from actions import dev_agent, desktop
    with pytest.raises(ValueError):
        dev_agent._project_path(tmp_path, '../outside.py')
    with pytest.raises(ValueError):
        dev_agent._project_path(tmp_path, '.git/hooks/pre-commit')
    assert 'Unsupported run command' in dev_agent._run_project('powershell -Command anything', tmp_path)
    assert 'disabled' in desktop._execute_generated_code('print("never executed")')


def test_bounded_multi_step_plan_and_persisted_summary(setup):
    brain, provider, _, handler = setup
    provider.chat.side_effect = [
        {'ok': True, 'text': '{"type":"action","action":"web_search","args":{"query":"first"}}'},
        {'ok': True, 'text': '{"type":"action","action":"web_search","args":{"query":"second"}}'},
        {'ok': True, 'text': '{"type":"answer","text":"Both retrieved results compared."}'}]
    assert brain.process('Research first and then compare with second')['text'] == 'Both retrieved results compared.'
    assert handler.call_count == 2
    assert brain.memory.recent_conversation()[0]['response'] == 'Both retrieved results compared.'


def test_local_wake_audio_uses_existing_detector():
    from tests.test_runtime_mode import adapter
    import numpy as np
    app = adapter(['_observe_local_audio'], {'np': np})
    app._wake_enabled = True
    app._awake = False
    app._wake_detector = Mock()
    app.ui = Mock(muted=False)
    app._observe_local_audio(b'\x00\x00' * 1600)
    app._wake_detector.feed.assert_called_once()
    app._awake = True
    app._observe_local_audio(b'\x00\x00' * 1600)
    assert app._wake_detector.feed.call_count == 1


def test_remote_pcm_uses_same_local_queue_and_privacy_gate():
    from core.offline_voice import OfflineVoice
    voice = OfflineVoice({}, Mock(), lambda: True, Mock(), Mock())
    voice.ready = True
    voice.submit_remote(b'\x01\x00' * 1600)
    assert voice.chunks.get_nowait() == b'\x01\x00' * 1600
    voice.allowed = lambda: False
    voice.submit_remote(b'\x01\x00' * 1600)
    assert voice.chunks.empty()


@pytest.mark.parametrize('uploaded', [False, True])
def test_private_configuration_cannot_be_read_as_a_tool(setup, uploaded):
    brain, provider, registry, handler = setup
    registry._actions['file_processor'] = ActionRecord('file_processor', 'Process files',
        {'type': 'OBJECT', 'properties': {'file_path': {'type': 'STRING'}}, 'required': ['file_path']},
        handler, valid=True)
    args = {} if uploaded else {'file_path': 'config/api_keys.json'}
    brain.action_planner.context_reader = lambda: {'current_file': 'config/api_keys.json'}
    provider.chat.side_effect = [{'ok': True, 'text': json.dumps({'type': 'action', 'action': 'file_processor', 'args': args})}]
    assert brain.process('Read the uploaded file')['intent'] == 'invalid_action'
    handler.assert_not_called()


def test_audio_settings_restart_local_audio_without_replacing_memory():
    from tests.test_runtime_mode import adapter
    app = adapter(['_run_local'])
    first, second = Mock(active=False), Mock(active=False)
    app._create_local_voice = Mock(side_effect=[first, second])
    app._runtime = SimpleNamespace(offline=True)
    app._offline_brain = object()
    original_brain = app._offline_brain
    app.ui = Mock()
    app._dashboard = None
    app._wake_enabled = False
    app._local_busy = 0
    app._is_speaking = False
    app._cancel_local_responses = Mock()
    async def wait():
        await asyncio.Event().wait()
    app._local_services = app._run_sleep_watch = wait
    async def exercise():
        app._reconnect_event = asyncio.Event()
        task = asyncio.create_task(app._run_local())
        await asyncio.sleep(.01)
        app._reconnect_event.set()
        for _ in range(100):
            if second.start.called:
                break
            await asyncio.sleep(.01)
        first.close.assert_called_once()
        second.start.assert_called_once()
        assert app._offline_brain is original_brain
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(exercise())
