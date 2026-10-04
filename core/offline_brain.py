"""Local-first arbitration. Learned text is never an execution plan."""
from dataclasses import dataclass
from threading import RLock

from core.brain.confidence import clamp
from core.brain.intent_engine import normalize, similarity
from core.learning.learner import Learner
from core.learning.knowledge_extractor import contains_secret, unsafe_to_learn
from core.providers.ollama_provider import OllamaProvider
from memory.alpha_memory import AlphaMemory
from memory.config_manager import get_offline_brain_settings


@dataclass
class Command:
    phrase: str
    handler: object
    intent: str
    dangerous: bool = False


class OfflineBrain:
    def __init__(self, memory=None, provider=None, settings=None, skill_matcher=None,
                 context_matcher=None, confirmer=None):
        self.settings = settings if settings is not None else get_offline_brain_settings()
        self.memory = memory if memory is not None else AlphaMemory()
        self.provider = provider if provider is not None else OllamaProvider(
            self.settings.get('ollama_url', 'http://127.0.0.1:11434'),
            self.settings.get('ollama_model', ''), self.settings.get('ollama_enabled', True))
        self.learner = Learner(self.memory)
        self.commands = {}
        self.skill_matcher = skill_matcher
        self.context_matcher = context_matcher
        self.confirmer = confirmer
        self._lock = RLock()

    def register_command(self, phrase, handler, intent='known_command', dangerous=False):
        self.commands[normalize(phrase)] = Command(phrase, handler, intent, dangerous)

    @staticmethod
    def result(text, source='local', confidence=0.0, intent='unknown', **extra):
        return dict(text=str(text), source=source, confidence=clamp(confidence), intent=intent, learned=False, **extra)

    def _execute(self, command, confidence, context, exact=False):
        # Non-exact action matches always need a gate; confidence alone is not permission.
        if command.dangerous or not exact:
            if self.confirmer is None:
                return self.result('This action needs explicit confirmation.', confidence=confidence,
                                   intent=command.intent, confirmation_required=True)
            text = self.confirmer(command.intent, command.phrase,
                                  lambda: command.handler(context))
            return self.result(text, confidence=confidence, intent=command.intent, confirmation_required=True)
        try:
            return self.result(command.handler(context), confidence=confidence, intent=command.intent)
        except Exception:
            return self.result('The local action could not complete.', intent='action_error')

    def _threshold(self):
        return max(0.80, clamp(self.settings.get('local_threshold', 0.80)))

    def lookup_local(self, text, context=None):
        """No network; None means the legacy pipeline may continue unchanged."""
        context = context or {}
        query = normalize(text)
        if not query:
            return None
        command = self.commands.get(query)
        if command:
            return self._execute(command, 1.0, context, exact=True)
        if self.skill_matcher:
            # Hook returns an explicitly registered Command and confidence, never generated code.
            match = self.skill_matcher(text, context)
            if match and clamp(match[1]) >= self._threshold():
                return self._execute(match[0], match[1], context, exact=True)
        alias = self.memory.get_alias(text)
        if alias and alias['confidence'] >= self._threshold():
            command = self.commands.get(normalize(alias['target']))
            if command:
                return self._execute(command, alias['confidence'], context)
        learned = self.memory.search(text, fuzzy=False)
        if learned and learned['confidence'] >= self._threshold() and not unsafe_to_learn(learned['response']):
            self.memory.mark_used(learned['id'])
            return self.result(learned['response'], 'memory', learned['confidence'], 'learned_solution', knowledge_id=learned['id'])
        if self.context_matcher:
            match = self.context_matcher(text, context)
            if match and clamp(match[1]) >= self._threshold():
                return self.result(match[0], confidence=match[1], intent='context')
        learned = self.memory.search(text)
        if learned and learned['confidence'] >= self._threshold() and not unsafe_to_learn(learned['response']):
            self.memory.mark_used(learned['id'])
            return self.result(learned['response'], 'memory', learned['confidence'], 'learned_solution', knowledge_id=learned['id'])
        candidates = [(similarity(query, key), command) for key, command in self.commands.items()]
        if candidates:
            score, command = max(candidates, key=lambda pair: pair[0])
            if score >= self._threshold():
                return self._execute(command, score, context)
        return None

    def process(self, text, context=None):
        with self._lock:
            if not self.settings.get('enabled', True):
                return self.result('Offline brain is disabled.', intent='disabled')
            if not isinstance(text, str) or not text.strip():
                return self.result('Please enter a request.', intent='empty')
            local = self.lookup_local(text, context)
            if local:
                return local
            if contains_secret(text) or (context and contains_secret(context.get('summary', ''))):
                return self.result('Sensitive input is not sent to the fallback or stored.', intent='sensitive_input')
            if not self.settings.get('ollama_enabled', True):
                return self.result('I do not have a confident local answer. Ollama fallback is disabled.')
            answer = self.provider.generate(text, context)
            if not answer['ok']:
                return self.result('I do not have a confident local answer. ' + answer['text'])
            result = self.result(answer['text'], 'ollama', 0.60, 'teacher_answer')
            if self.settings.get('auto_learn', False):
                knowledge_id = self.learner.learn_from_solution(text, answer['text'])
                result['learned'] = knowledge_id is not None
                result['knowledge_id'] = knowledge_id
            return result

    def startup_report(self):
        status = self.provider.is_available() if self.settings.get('ollama_enabled', True) else False
        return ['[ALPHA BRAIN] Offline core initialized', '[MEMORY] SQLite memory ready',
                '[OLLAMA] Available: ' + ('yes' if status else 'no'), '[ALPHA BRAIN] Mode: offline-first']
