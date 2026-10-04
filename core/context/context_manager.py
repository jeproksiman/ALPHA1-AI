from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from core.learning.knowledge_extractor import contains_secret


@dataclass
class Interaction:
    user_input: str
    result: dict
    id: str = field(default_factory=lambda: uuid4().hex)
    at: float = field(default_factory=monotonic)
    feedback: set = field(default_factory=set)


class ContextManager:
    def __init__(self, limit=20, max_age=900):
        self.history = deque(maxlen=max(1, min(limit, 100)))
        self.max_age = max_age
        self.entities = deque(maxlen=10)

    def remember(self, text, result):
        if len(text) > 500 or len(str(result.get('text', ''))) > 2000 or len(str(result))>6000:
            return None
        if contains_secret(text) or contains_secret(str(result)):
            return None
        interaction = Interaction(text, deepcopy(result))
        self.history.append(interaction)
        entity = result.get('entity')
        if entity and result.get('success') is True and not result.get('confirmation_required'):
            if entity not in self.entities:
                self.entities.append(deepcopy(entity))
        return interaction

    def recent(self, feedback=False):
        for item in reversed(self.history):
            if monotonic() - item.at > self.max_age:
                continue
            if feedback and (item.result.get('intent') in ('unknown', 'action_error', 'disabled', 'sensitive_input')
                             or item.result.get('confirmation_required')):
                return None
            return item
        return None

    def repeatable(self):
        for item in reversed(self.history):
            if monotonic() - item.at <= self.max_age and item.result.get('command_phrase'):
                if item.result.get('success') is True and not item.result.get('confirmation_required'):
                    return item
                return None
        return None

    def resolve_entity(self, kind=None):
        recent = self.recent()
        if not recent:
            return None
        entity = recent.result.get('entity')
        if entity and recent.result.get('success') is True and not recent.result.get('confirmation_required') and (kind is None or entity['kind'] == kind):
            return deepcopy(entity)
        return None

    def summary(self):
        item = self.recent()
        if not item:
            return ''
        return f"Recent topic: {item.user_input[:300]}\nRecent answer: {item.result['text'][:700]}"
