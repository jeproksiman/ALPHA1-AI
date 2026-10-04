"""Local diagnostics and conservative maintenance; no automatic deletion."""
from datetime import datetime, timezone, timedelta
from core.brain.intent_engine import similarity


class MemoryConsolidator:
    def __init__(self, memory):
        self.memory = memory

    def analyze(self):
        rows = self.memory.inspect(limit=50)
        pairs = []
        for i, left in enumerate(rows):
            for right in rows[i+1:]:
                if similarity(left['trigger_text'], right['trigger_text']) >= 0.80:
                    pairs.append({'ids':[left['id'],right['id']], 'kind':
                                  'near_duplicate' if left['response'] == right['response'] else 'conflict'})
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).replace(tzinfo=None).isoformat(sep=' ')
        return {'pairs':pairs, 'conflicts':self.memory.inspect('conflicts'),
                'stale_candidates':[row['id'] for row in rows if row['verification_status']=='candidate' and row['updated_at']<cutoff],
                'frequently_successful':[row['id'] for row in rows if row['success_count']>=3],
                'exact_duplicates':'Prevented by the normalized-trigger unique constraint.'}

    def deduplicate_candidates(self):
        # Preserve diagnostic records; only disable identical near-duplicate candidates.
        count = 0
        with self.memory._lock, self.memory.db:
            rows = self.memory.inspect(limit=50)
            for i, left in enumerate(rows):
                if left['verification_status'] != 'candidate':
                    continue
                for right in rows[i+1:]:
                    if right['verification_status']=='candidate' and left['response']==right['response'] and similarity(left['trigger_text'],right['trigger_text'])>=0.80:
                        self.memory.db.execute("UPDATE learned_knowledge SET verification_status='rejected',last_feedback='duplicate candidate' WHERE id=?", (left['id'],))
                        count += 1
                        break
        return count

    def trust_successful(self):
        with self.memory._lock, self.memory.db:
            return self.memory.db.execute('''UPDATE learned_knowledge SET verification_status='trusted'
                WHERE verification_status='verified' AND success_count>=3 AND confidence>=0.85''').rowcount
