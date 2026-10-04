"""Exact feedback grammar, with per-interaction receipts to prevent inflation."""
import re
from core.brain.intent_engine import normalize


class FeedbackEngine:
    POSITIVE = {'that worked', 'it worked', 'perfect', 'that fixed it'}
    NEGATIVE = {'that didn t work', 'that didnt work', 'wrong', 'that was wrong', 'it failed', 'don t use that solution', 'dont use that solution'}
    REMEMBER = {'remember this', 'remember how we fixed this', 'learn this solution'}

    def __init__(self, brain):
        self.brain = brain

    def handle(self, text):
        text = re.sub(r'^alpha[, ]+','',text.strip(),flags=re.I)
        query = normalize(text)
        correction = re.fullmatch(r'(?:no[, ]+)?(?:the correct (?:answer|solution) is|instead use)\s+(.+)', text.strip(), re.I | re.S)
        if query not in self.POSITIVE | self.NEGATIVE | self.REMEMBER and not correction:
            return None
        item = self.brain.context.recent(feedback=True)
        if not item:
            return self.brain.result('Which recent answer or action do you mean?', intent='feedback')
        kind = 'correction' if correction else 'success' if query in self.POSITIVE else 'failure' if query in self.NEGATIVE else 'remember'
        if kind in item.feedback:
            return self.brain.result('That feedback has already been recorded for this interaction.', intent='feedback')
        knowledge_id = item.result.get('knowledge_id')
        extracted = self.brain.learner.extractor.extract(item.user_input,item.result['text'])
        comparable = extracted['response'] if extracted else item.result['text']
        if item.result.get('command_phrase'):
            if kind == 'remember':
                item.feedback.add(kind)
                return self.brain.result('This registered action is already available locally. You can give it an alias or routine.', intent='feedback')
            # Actions have no executable learned-knowledge entry. Only record bounded diagnostics.
            if kind in ('success', 'failure'):
                item.feedback.add(kind)
                self.brain.memory.record_interaction(item.user_input, item.result, kind == 'success')
                if kind == 'failure':
                    item.result['success'] = False
                    if item.result.get('routine_id'):
                        self.brain.routines.feedback(item.result['routine_id'],False)
                return self.brain.result('Action feedback recorded.', intent='feedback')
            return self.brain.result('Use an explicit alias to correct an application or project target.', intent='feedback')
        if kind == 'failure' and not knowledge_id:
            existing = self.brain.memory.search(item.user_input,fuzzy=False)
            if existing and existing['response'] == comparable:
                knowledge_id = existing['id']
                item.result['knowledge_id'] = knowledge_id
            # Explicit negative feedback permits filtered failure diagnostics, never reuse.
        if not knowledge_id:
            knowledge_id = self.brain.learner.learn_from_solution(item.user_input, item.result['text'], source='ollama')
            item.result['knowledge_id'] = knowledge_id
        if not knowledge_id:
            return self.brain.result('That response is not suitable for long-term learning.', intent='feedback')
        stored = self.brain.memory.get_knowledge(knowledge_id)
        # A duplicate candidate can have a different proposed answer. Never verify the old
        # response using feedback on the new conflicting answer.
        if stored['response'] != comparable:
            return self.brain.result('This answer conflicts with saved knowledge. Use "the correct solution is ..." to replace it explicitly.', intent='feedback') if not correction else self._correct(item, knowledge_id, correction.group(1))
        if correction:
            return self._correct(item, knowledge_id, correction.group(1))
        if kind in ('success', 'failure'):
            veto = query in {'don t use that solution','dont use that solution'}
            changed = self.brain.memory.record_feedback(knowledge_id, kind == 'success', event_key=item.id + ':' + kind,
                reject=veto,reason='explicit do not reuse' if veto else None)
            if not changed:
                return self.brain.result('Feedback could not change this memory.', intent='feedback')
        item.feedback.add(kind)
        state = self.brain.memory.get_knowledge(knowledge_id)
        return self.brain.result(f"Solution saved as {state['verification_status']} (confidence {state['confidence']:.2f}).",
                                 intent='feedback', knowledge_id=knowledge_id)

    def _correct(self, item, knowledge_id, response):
        if not self.brain.memory.replace_solution(knowledge_id, response.strip()):
            return self.brain.result('That correction is not suitable for storage.', intent='feedback')
        item.feedback.add('correction')
        item.result['text'] = response.strip()
        return self.brain.result('Correction saved; the previous answer is preserved in correction history.', intent='feedback', knowledge_id=knowledge_id)
