"""SQLite solution store. Existing personal JSON memory remains independent."""
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import RLock

from core.brain.intent_engine import normalize, knowledge_similarity
from core.brain.confidence import clamp
from core.learning.knowledge_extractor import unsafe_to_learn
from memory.config_manager import BASE_DIR


class AlphaMemory:
    def __init__(self, path=None):
        path = Path(path) if path is not None else BASE_DIR / "memory" / "alpha_memory.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = RLock()
        self.db = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.db:
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS learned_knowledge (
                    id INTEGER PRIMARY KEY, topic TEXT NOT NULL,
                    trigger_text TEXT NOT NULL, normalized_trigger TEXT NOT NULL UNIQUE,
                    response TEXT NOT NULL, solution TEXT NOT NULL, source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, last_used_at TEXT);
                CREATE TABLE IF NOT EXISTS aliases (
                    id INTEGER PRIMARY KEY, phrase TEXT NOT NULL UNIQUE,
                    target TEXT NOT NULL, confidence REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY, original_input TEXT NOT NULL,
                    wrong_result TEXT NOT NULL, corrected_result TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY, user_input TEXT NOT NULL, response TEXT NOT NULL,
                    source TEXT NOT NULL, intent TEXT, confidence REAL NOT NULL, success INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE INDEX IF NOT EXISTS knowledge_confidence ON learned_knowledge(confidence);
            ''')
        self._migrate()

    def _migrate(self):
        """Additive Phase 2 migration. Never delete or rebuild a user database."""
        with self._lock, self.db:
            columns = {row[1] for row in self.db.execute('PRAGMA table_info(learned_knowledge)')}
            additions = {
                'verification_status': "TEXT NOT NULL DEFAULT 'candidate'",
                'use_count': 'INTEGER NOT NULL DEFAULT 0',
                'last_feedback': 'TEXT', 'last_feedback_at': 'TEXT',
            }
            for name, declaration in additions.items():
                if name not in columns:
                    self.db.execute(f'ALTER TABLE learned_knowledge ADD COLUMN {name} {declaration}')
            if 'verification_status' not in columns:
                self.db.execute('''UPDATE learned_knowledge SET verification_status=CASE
                    WHEN failure_count>=2 THEN 'rejected'
                    WHEN failure_count>0 THEN 'candidate'
                    WHEN success_count>=3 AND confidence>=0.85 THEN 'trusted'
                    WHEN success_count>0 OR (source!='ollama' AND confidence>=0.8) THEN 'verified'
                    ELSE 'candidate' END''')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS confidence_events (
                    id INTEGER PRIMARY KEY, knowledge_id INTEGER NOT NULL,
                    delta REAL NOT NULL, reason TEXT NOT NULL, event_key TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS routines (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, trigger_phrase TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1, success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS routine_steps (
                    id INTEGER PRIMARY KEY, routine_id INTEGER NOT NULL,
                    step_order INTEGER NOT NULL, action_name TEXT NOT NULL, action_args TEXT NOT NULL,
                    UNIQUE(routine_id,step_order));
                CREATE TABLE IF NOT EXISTS knowledge_conflicts (
                    id INTEGER PRIMARY KEY, knowledge_id INTEGER NOT NULL,
                    proposed_response TEXT NOT NULL, source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(knowledge_id,proposed_response));
                CREATE INDEX IF NOT EXISTS knowledge_verification
                    ON learned_knowledge(verification_status,confidence);
                PRAGMA user_version=2;
            ''')

    def close(self):
        with self._lock:
            self.db.close()

    def store_solution(self, trigger_text, response, solution=None, topic="", source="ollama", confidence=0.60):
        confidence = min(clamp(confidence),0.70) if source=='ollama' else clamp(confidence)
        solution = solution or response
        values = (trigger_text, response, solution, topic, source)
        if any(unsafe_to_learn(v) for v in values) or not normalize(trigger_text):
            return None
        if len(trigger_text) > 500 or max(len(response), len(solution)) > 2000:
            return None
        with self._lock, self.db:
            # Deduplication must not promote an unverified candidate or overwrite feedback.
            self.db.execute('''INSERT INTO learned_knowledge
                (topic,trigger_text,normalized_trigger,response,solution,source,confidence,verification_status)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(normalized_trigger) DO NOTHING''',
                (topic, trigger_text, normalize(trigger_text), response, solution, source, clamp(confidence),
                 'verified' if source != 'ollama' and confidence >= 0.8 else 'candidate'))
            existing = self.db.execute('SELECT id,response FROM learned_knowledge WHERE normalized_trigger=?',
                                      (normalize(trigger_text),)).fetchone()
            if existing['response'] != response:
                self.db.execute('''INSERT OR IGNORE INTO knowledge_conflicts
                    (knowledge_id,proposed_response,source) VALUES (?,?,?)''', (existing['id'],response,source))
            return existing['id']

    def add_alias(self, phrase, target, confidence=0.95):
        if max(len(phrase),len(target))>500 or unsafe_to_learn(phrase + "\n" + target) or not normalize(phrase) or not normalize(target):
            return False
        with self._lock, self.db:
            self.db.execute('''INSERT INTO aliases(phrase,target,confidence) VALUES (?,?,?)
                ON CONFLICT(phrase) DO UPDATE SET target=excluded.target,
                confidence=excluded.confidence,updated_at=CURRENT_TIMESTAMP''',
                (normalize(phrase), target, clamp(confidence)))
        return True

    def get_alias(self, phrase):
        with self._lock:
            row = self.db.execute('SELECT * FROM aliases WHERE phrase=?', (normalize(phrase),)).fetchone()
            return dict(row) if row else None

    def search(self, text, fuzzy=True, eligible_only=False):
        alias = self.get_alias(text)
        query = normalize(alias['target'] if alias else text)
        with self._lock:
            condition = " AND verification_status IN ('verified','trusted')" if eligible_only else ''
            row = self.db.execute('SELECT * FROM learned_knowledge WHERE normalized_trigger=?' + condition, (query,)).fetchone()
            if row:
                result = dict(row)
                result = self._rank(result, 1.0, alias)
                return result
            if not fuzzy:
                return None
            rows = self.db.execute('SELECT * FROM learned_knowledge WHERE 1=1' + condition
                                  + ' ORDER BY confidence DESC LIMIT 1000').fetchall()
        best = None
        for row in rows:
            result = dict(row)
            result = self._rank(result, knowledge_similarity(query, result['normalized_trigger']), alias)
            if result['confidence'] > 0 and (best is None or result['confidence'] > best['confidence']):
                best = result
        return best

    @staticmethod
    def _rank(result, match, alias=None):
        stored = result['confidence']
        verified = result['verification_status'] in ('verified', 'trusted')
        bonus = (0.08 if verified else 0) + (0.03 if result['verification_status'] == 'trusted' else 0)
        bonus += min(result['success_count'], 3) * 0.01 + min(result['use_count'], 5) * 0.002
        try:
            age = datetime.now(timezone.utc).replace(tzinfo=None) - datetime.fromisoformat(result['updated_at'])
            if timedelta(0) <= age <= timedelta(days=30) and verified:
                bonus += 0.01
        except ValueError:
            pass
        score = stored if match == 1 else clamp(match * min(1.0, stored + bonus))
        if result['verification_status'] == 'candidate':
            score = min(score, 0.70)
        if result['verification_status'] == 'rejected':
            score = 0.0
        if alias:
            score = min(score, alias['confidence'])
        result.update(stored_confidence=stored, confidence=score, match_score=match,
                      reason=f"{result['verification_status']} solution; match={match:.2f}, "
                             f"successes={result['success_count']}, failures={result['failure_count']}, "
                             f"uses={result['use_count']}, ranking bonus={bonus:.2f}, score={score:.2f}")
        return result

    def mark_used(self, knowledge_id):
        with self._lock, self.db:
            self.db.execute('''UPDATE learned_knowledge SET last_used_at=CURRENT_TIMESTAMP,
                use_count=use_count+1 WHERE id=?''', (knowledge_id,))

    def record_feedback(self, knowledge_id, success, reason=None, event_key=None, automatic=False, reject=False):
        with self._lock, self.db:
            row = self.db.execute('SELECT * FROM learned_knowledge WHERE id=?', (knowledge_id,)).fetchone()
            if not row:
                return False
            if event_key and self.db.execute('SELECT 1 FROM confidence_events WHERE event_key=?', (event_key,)).fetchone():
                return False
            if any(value and (len(value)>500 or unsafe_to_learn(value)) for value in (reason,event_key)):
                return False
            delta = (0.05 if automatic else 0.15) if success else (-0.10 if automatic else -0.20)
            updated = clamp(row['confidence'] + delta)
            successes = row['success_count'] + int(success)
            failures = row['failure_count'] + int(not success)
            status = row['verification_status']
            if not success:
                status = 'rejected' if reject or failures >= 2 else 'candidate'
            elif status != 'rejected' and not automatic:
                # A confirmed solution should answer its exact trigger after one success.
                updated = max(updated, 0.85)
                status = 'trusted' if successes >= 3 else 'verified'
            column = 'success_count' if success else 'failure_count'
            self.db.execute(f'''UPDATE learned_knowledge SET {column}={column}+1,
                confidence=?,verification_status=?,last_feedback=?,last_feedback_at=CURRENT_TIMESTAMP,
                updated_at=CURRENT_TIMESTAMP WHERE id=?''',
                (updated,status,'success' if success else 'failure',knowledge_id))
            event_reason = reason or ('automatic reuse' if automatic and success else 'automatic failure' if automatic
                                     else 'explicit success (verified confidence floor 0.85)' if success else 'explicit failure')
            self.db.execute('''INSERT INTO confidence_events(knowledge_id,delta,reason,event_key)
                VALUES (?,?,?,?)''', (knowledge_id,updated-row['confidence'],event_reason,event_key))
        return True

    def get_knowledge(self, knowledge_id):
        with self._lock:
            row = self.db.execute('SELECT * FROM learned_knowledge WHERE id=?', (knowledge_id,)).fetchone()
            return dict(row) if row else None

    def replace_solution(self, knowledge_id, response):
        if unsafe_to_learn(response) or not 12 <= len(response) <= 2000:
            return False
        with self._lock, self.db:
            row = self.get_knowledge(knowledge_id)
            if not row or not self.record_correction(row['trigger_text'], row['response'], response):
                return False
            self.db.execute('''UPDATE learned_knowledge SET response=?,solution=?,source='user',
                verification_status='verified',confidence=0.85,success_count=0,failure_count=0,
                updated_at=CURRENT_TIMESTAMP WHERE id=?''', (response,response,knowledge_id))
            self.db.execute('INSERT INTO confidence_events(knowledge_id,delta,reason) VALUES (?,?,?)',
                            (knowledge_id,0.85-row['confidence'],'explicit corrected replacement'))
        return True

    def inspect(self, kind='knowledge', topic=None, today=False, limit=10):
        limit = max(1, min(int(limit), 50))
        tables = {'knowledge':'learned_knowledge', 'aliases':'aliases', 'routines':'routines',
                  'events':'confidence_events', 'conflicts':'knowledge_conflicts'}
        table = tables[kind]
        with self._lock:
            rows = [dict(row) for row in self.db.execute(f'SELECT * FROM {table} ORDER BY id DESC LIMIT 1000')]
        if today:
            # User-facing date follows the configured Manila timezone, not host time.
            manila = timezone(timedelta(hours=8))
            now = datetime.now(manila).date()
            rows = [row for row in rows if datetime.fromisoformat(row['created_at']).replace(tzinfo=timezone.utc)
                    .astimezone(manila).date() == now]
        if topic:
            tokens = set(normalize(topic).split())
            rows = [row for row in rows if tokens & set(normalize(row.get('topic','')+' '+row.get('trigger_text','')).split())]
        return [row for row in rows if not unsafe_to_learn(json.dumps(row))][:limit]

    def counts(self):
        with self._lock:
            return {name:self.db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                    for name,table in [('knowledge','learned_knowledge'),('aliases','aliases'),('routines','routines')]}

    def semantic_candidates(self, limit=200):
        """Typed learned-solution adapter; personal profile memory stays separate."""
        with self._lock:
            rows = self.db.execute('''SELECT * FROM learned_knowledge
                WHERE verification_status IN ('verified','trusted')
                ORDER BY updated_at DESC,confidence DESC LIMIT ?''',(max(1,min(limit,500)),)).fetchall()
        return [dict(row) for row in rows if not unsafe_to_learn(json.dumps(dict(row)))]

    def record_correction(self, original_input, wrong_result, corrected_result):
        if any(unsafe_to_learn(v) or len(v) > 2000 for v in (original_input, wrong_result, corrected_result)):
            return False
        with self._lock, self.db:
            self.db.execute('INSERT INTO corrections(original_input,wrong_result,corrected_result) VALUES (?,?,?)',
                            (original_input, wrong_result, corrected_result))
        return True

    def record_interaction(self, user_input, result, success=None):
        # Opt-in at the caller; no raw conversation logging by default.
        if any(unsafe_to_learn(v) or len(v) > 2000 for v in (user_input, result['text'],result['source'],result.get('intent') or '')):
            return False
        with self._lock, self.db:
            self.db.execute('''INSERT INTO interaction_history
                (user_input,response,source,intent,confidence,success) VALUES (?,?,?,?,?,?)''',
                (user_input,result['text'],result['source'],result.get('intent'),clamp(result['confidence']),success))
        return True
