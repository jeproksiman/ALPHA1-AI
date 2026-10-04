"""Deterministic learning/control grammar. Does not accept generated shell code."""
import re
from pathlib import Path
from core.brain.intent_engine import normalize
from core.learning.knowledge_extractor import unsafe_to_learn


class AdaptiveRouter:
    def __init__(self, brain):
        self.brain = brain
        self.project_factory = None

    def resolve_command(self, phrase, seen=None):
        query = normalize(phrase)
        raw = re.fullmatch(r'(open|launch|start|run)\s+(.+)',phrase.strip(),re.I)
        if raw and self.project_factory and Path(raw[2].strip('"')).is_absolute():
            return self.project_factory(raw[1].lower(),raw[2].strip('"'))
        if query in self.brain.commands:
            return self.brain.commands[query]
        seen = set() if seen is None else seen
        if query in seen or len(seen)>5:
            return None
        seen.add(query)
        alias = self.brain.memory.get_alias(query)
        if alias:
            target = alias['target']
        else:
            match = re.fullmatch(r'(open|launch|start|close|run) (.+)', query)
            if not match:
                return None
            alias = self.brain.memory.get_alias(match[2].removeprefix('my '))
            if not alias:
                return None
            target = alias['target']
            if target.startswith('project:'):
                return self.project_factory(match[1], target[8:]) if self.project_factory else None
            # Replace only the action verb; app aliases retain a canonical registered target.
            target = re.sub(r'^(open|launch|start|close|run)\s+', '', target)
            target = match[1]+' '+target
        if target.startswith('project:'):
            return self.project_factory('open',target[8:]) if self.project_factory else None
        return self.resolve_command(target, seen)

    def handle(self, text, include_context=True):
        clean = re.sub(r'^alpha[, ]+','',text.strip(),flags=re.I).strip().rstrip('.?!')
        query = normalize(clean)
        inspection = self._inspect(clean,query)
        if inspection:
            return inspection
        # Routines require a colon and a list of explicit registered phrases.
        routine = re.fullmatch(r'when i say ([^,:]{1,100}):\s*(.+)',clean,re.I|re.S)
        if routine:
            steps = [step.strip().rstrip('.') for step in re.split(r',|\n|\s+then\s+',routine[2]) if step.strip()]
            routine_id = self.brain.routines.save(routine[1].strip(),steps)
            return self.brain.result('Routine saved. Say its name to request execution.' if routine_id else
                'Routine not saved. Use only registered, permitted actions with explicit targets.',intent='routine_learning')
        alias = re.fullmatch(r'when i say (.+?)[, ]+(?:i mean|open|use)\s+(.+)',clean,re.I)
        correction = re.fullmatch(r'no[, ]+by (.+?) i mean (.+)',clean,re.I)
        if alias or correction:
            match = alias or correction
            return self._alias(match[1].strip(), match[2].strip())
        command = self.resolve_command(clean)
        if command and normalize(command.phrase) != query:
            return self.brain._execute(command,0.98,{},exact=False)
        routine = self.brain.routines.get(query)
        if routine:
            return self._routine(query)
        if query.startswith(('when i say','no by')):
            return self.brain.result('Please give an explicit alias or a colon-separated routine.',intent='clarification')
        return self.recent(query) if include_context else None

    def recent(self, query):
        if query in {'again','do that again','repeat that','same thing'}:
            item = self.brain.context.repeatable()
            if not item:
                return self.brain.result('There is no recent successful action to repeat. Specify the action.',intent='clarification')
            phrase = item.result['command_phrase']
            if item.result.get('routine_id'):
                return self._routine(phrase)
            command = self.resolve_command(phrase)
            if not command or not command.repeatable:
                return self.brain.result('That action is not safely repeatable. Specify a new request.',intent='clarification')
            return self.brain._execute(command,0.95,{},exact=True)
        if query in {'it','that','this','same one'}:
            entity = self.brain.context.resolve_entity()
            return self.brain.result('Do you mean '+entity['target']+'? Specify the action.' if entity else
                'Which application, project, or answer do you mean?',intent='clarification')
        contextual = re.fullmatch(r'(close|run|open) (?:it|that|this|same one)',query)
        if contextual:
            entity = self.brain.context.resolve_entity('project' if contextual[1]=='run' else None)
            if not entity:
                return self.brain.result('Which application or project do you mean?',intent='clarification')
            if entity['kind']=='project':
                command = self.project_factory(contextual[1],entity['target']) if self.project_factory else None
            else:
                command = self.resolve_command(contextual[1]+' '+entity['target'])
            if not command:
                return self.brain.result('That contextual action is not registered. Please specify a supported action.',intent='clarification')
            # Pronouns always pass the existing gate, even when the entity is recent.
            return self.brain._execute(command,0.90,{},exact=False)
        return None

    def _alias(self, phrase, target):
        if unsafe_to_learn(phrase+'\n'+target):
            return self.brain.result('That alias is not suitable for storage.',intent='alias_learning')
        # Project aliases designate local directories, never executable code.
        raw_path = Path(target.strip('"'))
        if raw_path.is_absolute():
            if target.startswith(('\\\\','//')) or not raw_path.is_dir():
                return self.brain.result('Use an existing local project directory.',intent='alias_learning')
            canonical = 'project:'+str(raw_path.resolve())
        else:
            command = self.resolve_command(target) or self.resolve_command('open '+target)
            if not command:
                return self.brain.result('I do not have a registered target for that alias. Name a known application or local project directory.',intent='alias_learning')
            canonical = command.phrase
        if self.brain.memory.add_alias(phrase,canonical,0.98):
            return self.brain.result('Alias saved locally.',intent='alias_learning')
        return self.brain.result('Alias could not be saved.',intent='alias_learning')

    def _routine(self, phrase):
        routine = self.brain.routines.get(phrase)
        if not routine or not routine['enabled']:
            return self.brain.result('Routine is disabled.',intent='clarification')
        if self.brain.routines.preflight(routine) is None:
            return self.brain.result('Routine needs review because its saved steps are invalid or changed.',intent='clarification')
        if self.brain.confirmer is None:
            return self.brain.result('This routine needs explicit confirmation.',intent='routine',confirmation_required=True)
        def run():
            with self.brain._lock:
                current = self.brain.routines.get(phrase)
                if not current or current['id'] != routine['id'] or current['steps'] != routine['steps']:
                    return 'Routine changed after confirmation was requested. Review it and request execution again.'
                result = self.brain.routines.execute(phrase,lambda command:
                    self.brain._execute(command,1.0,{},exact=True,confirmed=True))
                routed = self.brain.result(result['text'],intent='routine',success=result['success'],
                                          command_phrase=phrase,routine_id=routine['id'],repeatable=True)
                self.brain.context.remember(phrase,routed)
                return routed['text']
        import json
        title = routine['name'] + ': ' + ', '.join(json.loads(step['action_args'])['command_phrase'] for step in routine['steps'])
        message = self.brain.confirmer('routine',title,run)
        return self.brain.result(message,intent='routine',confirmation_required=True)

    def _inspect(self, clean, query):
        kind = None
        if query in {'what did you learn today','what have you learned today'}:
            rows = self.brain.memory.inspect(today=True)
            kind = 'knowledge'
        elif query in {'show your recent memories','show recent memories'}:
            rows = self.brain.memory.inspect()
            kind = 'knowledge'
        elif query in {'what aliases have i taught you','what aliases do you know'}:
            rows = self.brain.memory.inspect('aliases')
            kind = 'aliases'
        elif query in {'what routines do you know','show routines'}:
            rows = self.brain.memory.inspect('routines')
            kind = 'routines'
        elif query in {'why did you choose that solution','why that solution'}:
            item = self.brain.context.recent()
            return self.brain.result(item.result.get('reason','The recent result came from an explicit local match or the reasoning fallback.') if item else 'There is no recent result to explain.',intent='inspection')
        elif query in {'suggest a routine','show workflow suggestions'}:
            suggestions = self.brain.patterns.suggestions()
            return self.brain.result('\n'.join('You frequently use '+', '.join(item['steps'])+'. To save it, explicitly say "When I say NAME: '+', '.join(item['steps'])+'".' for item in suggestions) or 'No repeated workflow is ready to suggest.',intent='pattern_suggestion')
        else:
            match = re.fullmatch(r'what do you remember about (.+)',clean,re.I)
            if match:
                rows = self.brain.memory.inspect(topic=match[1])
                kind = 'knowledge'
        if kind is None:
            return None
        if kind == 'knowledge':
            lines = [f"{row['trigger_text']}: {row['response']} ({row['verification_status']}, {row['confidence']:.2f})" for row in rows]
        elif kind == 'aliases':
            lines = [f"{row['phrase']} → {row['target']}" for row in rows]
        else:
            lines = [f"{row['name']} ({'enabled' if row['enabled'] else 'disabled'})" for row in rows]
        return self.brain.result('\n'.join(lines) or 'No matching local memories.',intent='inspection')
