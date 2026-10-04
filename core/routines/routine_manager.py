import json
from core.brain.intent_engine import normalize
from core.learning.knowledge_extractor import unsafe_to_learn


class RoutineManager:
    def __init__(self, memory, resolver):
        self.memory = memory
        self.resolver = resolver

    def save(self, name, steps):
        if not name or len(name)>100 or unsafe_to_learn(name) or not 1 <= len(steps) <= 10:
            return None
        commands = [self.resolver(step) for step in steps]
        if any(command is None or not command.action_name or not command.repeatable
               or not command.routine_allowed or unsafe_to_learn(command.phrase)
               or unsafe_to_learn(command.action_name)
               or unsafe_to_learn(json.dumps(command.action_args)) for command in commands):
            return None
        with self.memory._lock, self.memory.db:
            self.memory.db.execute('''INSERT INTO routines(name,trigger_phrase) VALUES (?,?)
                ON CONFLICT(trigger_phrase) DO UPDATE SET name=excluded.name,enabled=1,updated_at=CURRENT_TIMESTAMP''', (name,normalize(name)))
            routine_id = self.memory.db.execute('SELECT id FROM routines WHERE trigger_phrase=?', (normalize(name),)).fetchone()[0]
            self.memory.db.execute('DELETE FROM routine_steps WHERE routine_id=?', (routine_id,))
            for order, command in enumerate(commands):
                self.memory.db.execute('''INSERT INTO routine_steps(routine_id,step_order,action_name,action_args)
                    VALUES (?,?,?,?)''', (routine_id,order,command.action_name,
                    json.dumps({'command_phrase':command.phrase,'parameters':command.action_args},sort_keys=True)))
        return routine_id

    def get(self, phrase):
        with self.memory._lock:
            row = self.memory.db.execute('SELECT * FROM routines WHERE trigger_phrase=?', (normalize(phrase),)).fetchone()
            if not row:
                return None
            routine = dict(row)
            routine['steps'] = [dict(step) for step in self.memory.db.execute(
                'SELECT * FROM routine_steps WHERE routine_id=? ORDER BY step_order', (row['id'],))]
        return routine

    def execute(self, phrase, executor):
        routine = self.get(phrase)
        if not routine or not routine['enabled'] or not 1 <= len(routine['steps']) <= 10:
            return {'text':'Routine is unavailable or disabled.', 'success':False}
        # Preflight all steps before the first action, including after registry/config changes.
        commands = self.preflight(routine)
        if commands is None:
            return {'text':'Routine needs review because a registered action changed.', 'success':False}
        results = []
        for command in commands:
            result = executor(command)
            results.append(result['text'])
            if result.get('success') is not True or result.get('confirmation_required'):
                self.feedback(routine['id'], False)
                return {'text':'Routine stopped: '+result['text'], 'success':False}
        self.feedback(routine['id'], True)
        return {'text':'\n'.join(results), 'success':True}

    def preflight(self, routine):
        if unsafe_to_learn(routine['name']) or not 1 <= len(routine['steps']) <= 10:
            return None
        commands = []
        for step in routine['steps']:
            try:
                if unsafe_to_learn(step['action_name']+'\n'+step['action_args']):
                    return None
                args = json.loads(step['action_args'])
                command = self.resolver(args['command_phrase'])
                if command is None or not command.routine_allowed or not command.repeatable or command.action_name != step['action_name'] or command.action_args != args['parameters']:
                    raise ValueError('Routine changed')
                commands.append(command)
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
        return commands

    def feedback(self, routine_id, success):
        column = 'success_count' if success else 'failure_count'
        with self.memory._lock, self.memory.db:
            self.memory.db.execute(f'UPDATE routines SET {column}={column}+1,updated_at=CURRENT_TIMESTAMP WHERE id=?', (routine_id,))
