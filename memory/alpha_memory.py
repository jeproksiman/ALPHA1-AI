"""SQLite solution store. Existing personal JSON memory remains independent."""
import sqlite3
from pathlib import Path
from threading import RLock

from core.brain.intent_engine import normalize, similarity
from core.brain.confidence import clamp, feedback_confidence
from core.learning.knowledge_extractor import unsafe_to_learn
from memory.config_manager import BASE_DIR


class AlphaMemory:
    def __init__(self, path=None):
        path = Path(path) if path is not None else BASE_DIR / "memory" / "alpha_memory.db"
        path.parent.mkdir(parents=True, exist_ok=True)
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

    def close(self):
        with self._lock:
            self.db.close()

    def store_solution(self, trigger_text, response, solution=None, topic="", source="ollama", confidence=0.60):
        solution = solution or response
        values = (trigger_text, response, solution, topic, source)
        if any(unsafe_to_learn(v) for v in values) or not normalize(trigger_text):
            return None
        if len(trigger_text) > 500 or max(len(response), len(solution)) > 2000:
            return None
        with self._lock, self.db:
            # Deduplication must not promote an unverified candidate or overwrite feedback.
            self.db.execute('''INSERT INTO learned_knowledge
                (topic,trigger_text,normalized_trigger,response,solution,source,confidence)
                VALUES (?,?,?,?,?,?,?) ON CONFLICT(normalized_trigger) DO NOTHING''',
                (topic, trigger_text, normalize(trigger_text), response, solution, source, clamp(confidence)))
            return self.db.execute('SELECT id FROM learned_knowledge WHERE normalized_trigger=?',
                                   (normalize(trigger_text),)).fetchone()[0]

    def add_alias(self, phrase, target, confidence=0.95):
        if unsafe_to_learn(phrase + "\n" + target) or not normalize(phrase) or not normalize(target):
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

    def search(self, text, fuzzy=True):
        alias = self.get_alias(text)
        query = normalize(alias['target'] if alias else text)
        with self._lock:
            row = self.db.execute('SELECT * FROM learned_knowledge WHERE normalized_trigger=?', (query,)).fetchone()
            if row:
                result = dict(row)
                result['confidence'] = min(result['confidence'], alias['confidence']) if alias else result['confidence']
                return result
            if not fuzzy:
                return None
            rows = self.db.execute('SELECT * FROM learned_knowledge ORDER BY confidence DESC LIMIT 1000').fetchall()
        best = None
        for row in rows:
            result = dict(row)
            result['confidence'] *= similarity(query, result['normalized_trigger'])
            if alias:
                result['confidence'] = min(result['confidence'], alias['confidence'])
            if result['confidence'] > 0 and (best is None or result['confidence'] > best['confidence']):
                best = result
        return best

    def mark_used(self, knowledge_id):
        with self._lock, self.db:
            self.db.execute('UPDATE learned_knowledge SET last_used_at=CURRENT_TIMESTAMP WHERE id=?', (knowledge_id,))

    def record_feedback(self, knowledge_id, success):
        with self._lock, self.db:
            row = self.db.execute('SELECT confidence FROM learned_knowledge WHERE id=?', (knowledge_id,)).fetchone()
            if not row:
                return False
            column = 'success_count' if success else 'failure_count'
            self.db.execute(f'''UPDATE learned_knowledge SET {column}={column}+1,
                confidence=?,updated_at=CURRENT_TIMESTAMP WHERE id=?''',
                (feedback_confidence(row[0], success), knowledge_id))
        return True

    def record_correction(self, original_input, wrong_result, corrected_result):
        if any(unsafe_to_learn(v) or len(v) > 2000 for v in (original_input, wrong_result, corrected_result)):
            return False
        with self._lock, self.db:
            self.db.execute('INSERT INTO corrections(original_input,wrong_result,corrected_result) VALUES (?,?,?)',
                            (original_input, wrong_result, corrected_result))
        return True

    def record_interaction(self, user_input, result, success=None):
        # Opt-in at the caller; no raw conversation logging by default.
        if any(unsafe_to_learn(v) or len(v) > 2000 for v in (user_input, result['text'])):
            return False
        with self._lock, self.db:
            self.db.execute('''INSERT INTO interaction_history
                (user_input,response,source,intent,confidence,success) VALUES (?,?,?,?,?,?)''',
                (user_input,result['text'],result['source'],result.get('intent'),clamp(result['confidence']),success))
        return True
