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
from core.context.context_manager import ContextManager
from core.learning.feedback_engine import FeedbackEngine
from core.routines.pattern_detector import PatternDetector
from core.routines.routine_manager import RoutineManager


@dataclass
class Command:
    phrase: str
    handler: object
    intent: str
    dangerous: bool = False
    repeatable: bool = False
    routine_allowed: bool = False
    action_name: str = ''
    action_args: object = None
    entity: object = None


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
        self.context = ContextManager()
        self.patterns = PatternDetector()
        self.feedback = FeedbackEngine(self)
        from core.brain.adaptive_router import AdaptiveRouter
        self.adaptive = AdaptiveRouter(self)
        self.routines = RoutineManager(self.memory, self.adaptive.resolve_command)

    def register_command(self, phrase, handler, intent='known_command', dangerous=False, **metadata):
        self.commands[normalize(phrase)] = Command(phrase, handler, intent, dangerous, **metadata)

    @staticmethod
    def result(text, source='local', confidence=0.0, intent='unknown', **extra):
        return dict(text=str(text), source=source, confidence=clamp(confidence), intent=intent, learned=False, **extra)

    def _execute(self, command, confidence, context, exact=False, confirmed=False):
        if confirmed:
            context = dict(context, _alpha_confirmed=True)
        # Non-exact action matches always need a gate; confidence alone is not permission.
        if (command.dangerous or not exact) and not confirmed:
            if self.confirmer is None:
                return self.result('This action needs explicit confirmation.', confidence=confidence,
                                   intent=command.intent, confirmation_required=True)
            text = self.confirmer(command.intent, command.phrase,
                                  lambda: self._run_confirmed(command, confidence, context))
            return self.result(text, confidence=confidence, intent=command.intent, confirmation_required=True)
        try:
            output = command.handler(context)
            if isinstance(output, dict):
                result = self.result(output['text'], confidence=confidence, intent=command.intent,
                                     success=output.get('success', False))
            else:
                result = self.result(output, confidence=confidence, intent=command.intent, success=True)
            if command.action_name:
                result.update(command_phrase=command.phrase, action_name=command.action_name,
                              action_args=command.action_args, repeatable=command.repeatable, entity=command.entity)
                if result['success'] and command.repeatable and (not command.dangerous or confirmed):
                    self.patterns.record(command.phrase)
            return result
        except Exception:
            return self.result('The local action could not complete.', intent='action_error')

    def _run_confirmed(self, command, confidence, context):
        with self._lock:
            result = self._execute(command,confidence,context,exact=True,confirmed=True)
            self.context.remember(command.phrase,result)
            return result['text']

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
        learned = self.memory.search(text, fuzzy=False, eligible_only=True)
        if learned and learned['confidence'] >= self._threshold() and not unsafe_to_learn(learned['response']):
            self.memory.mark_used(learned['id'])
            return self.result(learned['response'], 'memory', learned['confidence'], 'learned_solution', knowledge_id=learned['id'], reason=learned['reason'])
        if self.context_matcher:
            match = self.context_matcher(text, context)
            if match and clamp(match[1]) >= self._threshold():
                return self.result(match[0], confidence=match[1], intent='context')
        learned = self.memory.search(text, eligible_only=True)
        if learned and learned['confidence'] >= self._threshold() and not unsafe_to_learn(learned['response']):
            self.memory.mark_used(learned['id'])
            return self.result(learned['response'], 'memory', learned['confidence'], 'learned_solution', knowledge_id=learned['id'], reason=learned['reason'])
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
            if contains_secret(text) or (context and contains_secret(context.get('summary', ''))):
                return self.result('Sensitive input is not sent to the fallback or stored.', intent='sensitive_input')
            feedback = self.feedback.handle(text)
            if feedback:
                return feedback
            adaptive = self.adaptive.handle(text)
            if adaptive:
                return self._finish(text,adaptive)
            local = self.lookup_local(text, context)
            if local:
                return self._finish(text,local)
            if not self.settings.get('ollama_enabled', True):
                return self._finish(text,self.result('I do not have a confident local answer. Ollama fallback is disabled.'))
            fallback_context = dict(context or {})
            fallback_context.setdefault('summary',self.context.summary())
            answer = self.provider.generate(text, fallback_context)
            if not answer['ok']:
                return self._finish(text,self.result('I do not have a confident local answer. ' + answer['text']))
            result = self.result(answer['text'], 'ollama', 0.60, 'teacher_answer')
            if contains_secret(answer['text']):
                return self._finish(text,self.result('The fallback response contained sensitive data and was discarded.',intent='sensitive_input'))
            if self.settings.get('auto_learn', False):
                knowledge_id = self.learner.learn_from_solution(text, answer['text'])
                result['learned'] = knowledge_id is not None
                result['knowledge_id'] = knowledge_id
            return self._finish(text,result)

    def _finish(self, text, result):
        # Control/inspection replies must not replace the answer being reviewed.
        if result['intent'] not in ('inspection','alias_learning','routine_learning','clarification','pattern_suggestion'):
            self.context.remember(text,result)
        return result

    def startup_report(self):
        status = self.provider.is_available() if self.settings.get('ollama_enabled', True) else False
        counts = self.memory.counts()
        return ['[ALPHA BRAIN] Offline intelligence ready',
                f"[MEMORY] {counts['knowledge']} learned items, {counts['aliases']} aliases, {counts['routines']} routines",
                '[OLLAMA] Available: ' + ('yes' if status else 'no'), '[LEARNING] Verified learning enabled']
