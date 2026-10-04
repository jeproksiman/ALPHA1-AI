import tempfile
import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from core.brain.app_commands import configure_app_commands
from core.brain.confidence import clamp, feedback_confidence
from core.learning.learner import Learner
from core.offline_brain import OfflineBrain, Command
from core.providers.ollama_provider import OllamaProvider
from memory.alpha_memory import AlphaMemory
from memory.config_manager import get_offline_brain_settings


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'memory.db'
        self.memory = AlphaMemory(self.path)
        self.provider = Mock()
        self.provider.generate.return_value = {'ok': True, 'text': 'A reusable answer from the local teacher.'}
        self.brain = OfflineBrain(self.memory, self.provider, settings={})

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_initialization_and_persistence(self):
        tables = {r[0] for r in self.memory.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'learned_knowledge', 'aliases', 'corrections', 'interaction_history',
                                  'confidence_events','knowledge_conflicts','routines','routine_steps','conversation_state','companion_facts'})
        self.memory.store_solution('question', 'A persistent local answer.', source='user',confidence=0.9)
        other = AlphaMemory(self.path)
        self.assertEqual(other.search('QUESTION!')['response'], 'A persistent local answer.')
        other.close()

    def test_memory_before_provider(self):
        item = self.memory.store_solution('what is sqlite', 'SQLite is a local relational database.', source='user',confidence=0.95)
        result = self.brain.process('What is SQLite?')
        self.assertEqual(result['source'], 'memory')
        self.assertEqual(result['knowledge_id'], item)
        self.provider.generate.assert_not_called()

    def test_alias_and_fuzzy(self):
        self.memory.store_solution('explain local database storage', 'Local storage stays on this computer.', source='user',confidence=1)
        self.memory.add_alias('describe storage', 'explain local database storage')
        self.assertEqual(self.brain.process('describe storage')['source'], 'memory')
        match = self.memory.search('explain local database storages')
        self.assertGreaterEqual(match['confidence'], 0.8)
        self.assertEqual(self.brain.process('explain local database storages')['source'], 'memory')

    def test_candidate_confidence_and_duplicate_feedback(self):
        learner = Learner(self.memory)
        item = learner.learn_from_solution('how does sqlite work', 'SQLite stores data locally in tables.', confidence=1)
        self.assertEqual(self.memory.search('how does sqlite work')['confidence'], 0.7)
        learner.record_success(item)
        learner.record_success(item)
        duplicate = learner.learn_from_solution('HOW DOES SQLITE WORK?', 'A conflicting replacement answer.')
        self.assertEqual(item, duplicate)
        self.assertAlmostEqual(self.memory.search('how does sqlite work')['confidence'], 1.0)
        self.assertEqual(self.memory.search('how does sqlite work')['success_count'], 2)
        learner.record_failure(item)
        self.assertEqual(self.memory.search('how does sqlite work')['verification_status'], 'candidate')
        self.assertIsNone(self.memory.search('how does sqlite work',eligible_only=True))

    def test_low_confidence_falls_back(self):
        self.memory.store_solution('question', 'This unverified answer is only a candidate.', confidence=0.60)
        self.assertEqual(self.brain.process('question')['source'], 'ollama')
        self.provider.generate.assert_called_once()

    def test_default_does_not_learn(self):
        result = self.brain.process('unknown question')
        self.assertFalse(result['learned'])
        self.assertIsNone(self.memory.search('unknown question'))

    def test_optional_learning(self):
        self.brain.settings['auto_learn'] = True
        result = self.brain.process('unknown question')
        self.assertTrue(result['learned'])
        self.assertEqual(self.memory.search('unknown question')['confidence'], 0.60)
        self.assertEqual(self.brain.process('unknown question')['source'], 'ollama')

    def test_secrets_rejected_at_all_write_boundaries(self):
        # Synthetic values only; no production credentials are read.
        secret = 'password=synthetic-test-value'
        self.assertIsNone(Learner(self.memory).learn_from_solution('question', secret))
        self.assertIsNone(self.memory.store_solution(secret, 'A reusable safe-looking answer.'))
        self.assertFalse(self.memory.add_alias(secret, 'question'))
        self.assertFalse(self.memory.record_correction('question', 'wrong', secret))
        self.assertFalse(self.memory.record_interaction(secret, self.brain.result('answer')))
        self.assertEqual(self.brain.process(secret)['intent'], 'sensitive_input')
        self.provider.generate.assert_not_called()

    def test_dangerous_learning_rejected(self):
        for solution in ('Remove-Item C:\\data -Recurse', 'format disk', 'disable security software',
                         'dump credentials', 'powershell -encodedcommand arbitrary'):
            self.assertIsNone(Learner(self.memory).learn_from_solution('how to do this', solution))

    def test_no_execution_from_learned_text(self):
        handler = Mock()
        self.brain.register_command('do action', handler)
        self.memory.store_solution('explain an action', 'do action', source='user',confidence=1)
        result = self.brain.process('explain an action')
        self.assertEqual(result['text'], 'do action')
        handler.assert_not_called()

    def test_action_confirmation_for_dangerous_alias_and_fuzzy(self):
        handler = Mock(return_value='Done')
        self.brain.register_command('open calculator', handler)
        self.memory.add_alias('use calculator', 'open calculator')
        self.assertTrue(self.brain.process('use calculator')['confirmation_required'])
        self.assertTrue(self.brain.process('open calculators')['confirmation_required'])
        self.brain.register_command('dangerous operation', handler, dangerous=True)
        self.assertTrue(self.brain.process('dangerous operation')['confirmation_required'])
        handler.assert_not_called()
        self.assertEqual(self.brain.process('open calculator')['text'], 'Done')
        handler.assert_called_once()

    def test_confirmation_hook_defers_handler(self):
        handler = Mock(return_value='Done')
        gate = Mock(return_value='Confirmation pending')
        self.brain.confirmer = gate
        self.brain.register_command('dangerous operation', handler, dangerous=True)
        self.brain.process('dangerous operation')
        handler.assert_not_called()
        self.assertEqual(gate.call_args.args[2](), 'Done')

    def test_hook_order_and_context(self):
        handler = Mock(return_value='Skill result')
        self.brain.skill_matcher = lambda text, ctx: (Command('skill', handler, 'skill'), 1)
        self.assertEqual(self.brain.process('skill')['text'], 'Skill result')
        self.brain.skill_matcher = None
        self.brain.context_matcher = lambda text, ctx: (ctx['answer'], 0.95)
        self.assertEqual(self.brain.process('context question', {'answer': 'Context answer'})['text'], 'Context answer')

    def test_app_adapter_restricts_targets(self):
        registry = Mock()
        configure_app_commands(self.brain, registry, ['calculator', 'powershell'])
        self.brain.process('launch calculator')
        registry.run.assert_called_once_with('open_app', {'app_name': 'calculator'}, {'player': None})
        registry.run.reset_mock()
        self.brain.process('open calculator & dangerous-command')
        registry.run.assert_not_called()
        self.brain.process('open powershell')
        registry.run.assert_not_called()

    def test_unavailable_and_disabled(self):
        self.provider.generate.return_value = {'ok': False, 'text': 'Ollama is unavailable.'}
        self.assertEqual(self.brain.process('unknown')['source'], 'local')
        self.brain.settings['ollama_enabled'] = False
        self.provider.reset_mock()
        self.brain.process('unknown')
        self.provider.generate.assert_not_called()
        self.brain.settings['enabled'] = False
        self.assertEqual(self.brain.process('unknown')['intent'], 'disabled')

    def test_confidence(self):
        self.assertEqual(clamp(2), 1)
        self.assertEqual(clamp(-1), 0)
        self.assertAlmostEqual(feedback_confidence(0.6, True), 0.7)
        with self.assertRaises(ValueError):
            clamp(float('nan'))

    @patch('memory.config_manager.load_api_keys', return_value={'llm_provider': 'openai', 'llm_url': 'http://localhost:1234'})
    def test_config_env_and_legacy_isolation(self, load):
        with patch.dict('os.environ', {'ALPHA_AUTO_LEARN_OLLAMA': 'false', 'ALPHA_LOCAL_CONFIDENCE_THRESHOLD': 'bad'}, clear=True):
            cfg = get_offline_brain_settings()
            self.assertEqual(cfg['ollama_url'], 'http://127.0.0.1:11434')
            self.assertEqual(cfg['local_threshold'], 0.8)
            self.assertFalse(cfg['auto_learn'])


class ProviderTests(unittest.TestCase):
    def session(self):
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {'models': [{'name': 'test:latest'}]}
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = {'done': True, 'message': {'content': 'Mock local answer'}}
        return session

    def test_mock_response_and_discovery(self):
        session = self.session()
        provider = OllamaProvider(session=session)
        self.assertEqual(provider.generate('question')['text'], 'Mock local answer')
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'test:latest')
        self.assertFalse(payload['stream'])
        self.assertNotIn('tools', payload)
        self.assertFalse(session.post.call_args.kwargs['allow_redirects'])
        self.assertFalse(session.trust_env)

    def test_unavailable_timeout_and_malformed(self):
        for error in (requests.ConnectionError(), requests.Timeout(), ValueError()):
            session = self.session()
            session.get.side_effect = error
            self.assertFalse(OllamaProvider(session=session).generate('question')['ok'])
        session = self.session()
        session.post.return_value.json.return_value = {'message': {'content': ''}}
        self.assertFalse(OllamaProvider(session=session).generate('question')['ok'])
        session.post.side_effect = requests.Timeout()
        self.assertFalse(OllamaProvider(session=session).generate('question')['ok'])

    def test_no_models_or_missing_configured_model(self):
        session = self.session()
        session.get.return_value.json.return_value = {'models': []}
        self.assertFalse(OllamaProvider(session=session).generate('question')['ok'])
        session = self.session()
        self.assertFalse(OllamaProvider(model='absent', session=session).generate('question')['ok'])
        session.post.assert_not_called()

    def test_invalid_remote_url_and_redirect(self):
        session = self.session()
        for url in ('https://example.com', 'http://example.com', 'http://user:pass@localhost:11434',
                    'http://localhost:11434?redirect=1'):
            self.assertFalse(OllamaProvider(url=url, session=session).is_available())
        session.get.assert_not_called()
        session.get.return_value.status_code = 302
        self.assertFalse(OllamaProvider(session=session).is_available())


class IntegrationTests(unittest.TestCase):
    @staticmethod
    def methods(path, class_name, names, namespace=None):
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        module = ast.Module(body=[ast.ClassDef(name='Adapter', bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
        ns = namespace or {}
        exec(compile(ast.fix_missing_locations(module), path, 'exec'), ns)
        return ns['Adapter']()

    def test_typed_callback_without_cloud_session(self):
        adapter = self.methods('main.py', 'JarvisLive', ['_on_offline_text_command'])
        adapter.ui = Mock()
        adapter._wake_enabled = False
        adapter._offline_brain = Mock()
        adapter._offline_brain.process.return_value = {'text': 'Local answer'}
        adapter._on_text_command = Mock()
        adapter._on_offline_text_command('question')
        adapter.ui.write_log.assert_called_once_with('ALPHA: Local answer')
        adapter._on_text_command.assert_not_called()
        adapter._on_offline_text_command('live: existing command')
        adapter._on_text_command.assert_called_once_with('existing command')

    def test_disabled_and_sleep_preserve_gates(self):
        adapter = self.methods('main.py', 'JarvisLive', ['_on_offline_text_command'])
        adapter.ui = Mock()
        adapter._wake_enabled = False
        adapter._offline_brain = None
        adapter._on_text_command = Mock()
        adapter._on_offline_text_command('existing command')
        adapter._on_text_command.assert_called_once_with('existing command')
        adapter._wake_enabled = True
        adapter._awake = False
        adapter._on_text_command.reset_mock()
        adapter._on_offline_text_command('existing command')
        adapter._on_text_command.assert_not_called()

    def test_only_input_box_uses_new_callback(self):
        # Regression guard: generated uploads/quiz/clipboard requests keep the old callback.
        tree = ast.parse(Path('ui.py').read_text(encoding='utf-8'))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(child, ast.Constant) and child.value == 'on_offline_text_command'
                for child in ast.walk(node)):
                names.append(node.name)
        self.assertEqual(set(names), {'_send', 'on_offline_text_command'})


if __name__ == '__main__':
    unittest.main()
