from core.learning.knowledge_extractor import KnowledgeExtractor
from core.brain.confidence import clamp


class Learner:
    def __init__(self, memory, extractor=None):
        self.memory = memory
        self.extractor = extractor or KnowledgeExtractor()

    def learn_from_solution(self, user_input, solution, source="ollama", confidence=0.60):
        extracted = self.extractor.extract(user_input, solution)
        if extracted is None:
            return None
        # Teacher output remains a candidate regardless of claimed confidence.
        confidence = min(clamp(confidence), 0.70) if source == 'ollama' else clamp(confidence)
        return self.memory.store_solution(**extracted, source=source, confidence=confidence)

    def record_success(self, knowledge_id):
        return self.memory.record_feedback(knowledge_id, True)

    def record_failure(self, knowledge_id):
        return self.memory.record_feedback(knowledge_id, False)

    def retrieve(self, text):
        return self.memory.search(text)
