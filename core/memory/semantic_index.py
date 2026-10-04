"""Lazy local embedding cache; SQLite and cosine only, no vector-stack dependency."""
import hashlib
import math
import sqlite3
import struct
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.learning.knowledge_extractor import unsafe_to_learn


def cosine(left, right):
    if len(left)!=len(right) or not left:
        return 0.0
    if any(not math.isfinite(v) for v in (*left,*right)):
        return 0.0
    norm = math.sqrt(sum(v*v for v in left)*sum(v*v for v in right))
    return max(-1.0,min(1.0,sum(a*b for a,b in zip(left,right))/norm)) if norm else 0.0


class SemanticIndex:
    def __init__(self, memory, provider, cache_path=None, enabled=True):
        self.memory = memory
        self.provider = provider
        self.enabled = enabled
        self.cache_path = Path(cache_path) if cache_path else memory.path.with_name('semantic_cache.db')

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _vectors(self, documents, model):
        # Open per lookup; no persistent handles and no startup database creation.
        self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.cache_path,timeout=2)) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL, namespace TEXT NOT NULL, item_key TEXT NOT NULL,
                content_hash TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(model,namespace,item_key))''')
            result = []
            missing = []
            for i,(namespace,key,text) in enumerate(documents):
                digest = self._digest(text)
                row = db.execute('''SELECT dimension,vector FROM embeddings
                    WHERE model=? AND namespace=? AND item_key=? AND content_hash=?''',
                    (model,namespace,key,digest)).fetchone()
                vector = None
                if row and 1<=row[0]<=8192 and len(row[1])==row[0]*4:
                    vector = list(struct.unpack('<'+'f'*row[0],row[1]))
                    if any(not math.isfinite(v) for v in vector) or not any(vector):
                        vector = None
                result.append(vector)
                if vector is None:
                    missing.append(i)
            for start in range(0,len(missing),16):
                indices = missing[start:start+16]
                response = self.provider.embed([documents[i][2] for i in indices])
                if not response['ok'] or len(response['embeddings'])!=len(indices):
                    return None
                for i,vector in zip(indices,response['embeddings']):
                    if not isinstance(vector,list) or not 1<=len(vector)<=8192 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in vector) or not any(vector):
                        return None
                    namespace,key,text = documents[i]
                    db.execute('''INSERT INTO embeddings(model,namespace,item_key,content_hash,dimension,vector)
                        VALUES (?,?,?,?,?,?) ON CONFLICT(model,namespace,item_key) DO UPDATE SET
                        content_hash=excluded.content_hash,dimension=excluded.dimension,
                        vector=excluded.vector,updated_at=CURRENT_TIMESTAMP''',
                        (model,namespace,key,self._digest(text),len(vector),struct.pack('<'+'f'*len(vector),*vector)))
                    result[i] = vector
            # Query vectors contain no raw text and retain only a bounded cache.
            db.execute('''DELETE FROM embeddings WHERE namespace='query' AND rowid NOT IN (
                SELECT rowid FROM embeddings WHERE namespace='query' ORDER BY updated_at DESC,rowid DESC LIMIT 100)''')
            db.commit()
            return result

    @staticmethod
    def rank(row, similarity):
        if row['verification_status'] not in ('verified','trusted') or similarity<0.80:
            return 0.0
        explicit = row['source']=='user'
        bonus = (0.05 if explicit else 0.0) + (0.02 if row['verification_status']=='trusted' else 0)
        bonus += min(row['success_count'],3)*0.005
        try:
            age = datetime.now(timezone.utc).replace(tzinfo=None)-datetime.fromisoformat(row['updated_at'])
            if timedelta(0)<=age<=timedelta(days=7):
                bonus += 0.02
        except ValueError:
            pass
        importance = max(0.0,min(1.0,float(row.get('importance',0.5))))
        return min(1.0,similarity*0.75+row['confidence']*0.25+bonus+importance*0.01)

    def search(self, text, threshold=0.80):
        if not self.enabled or not isinstance(text,str) or len(text)>500 or unsafe_to_learn(text):
            return None
        rows = self.memory.semantic_candidates()
        if not rows:
            return None
        model = getattr(self.provider,'embedding_model','')
        if not isinstance(model,str) or not model:
            return None
        documents = [('query',self._digest(text),text)]
        eligible = []
        for row in rows:
            document = row['trigger_text']+'\n'+row['response']
            if len(document)<=3000 and not unsafe_to_learn(document):
                documents.append(('learned_solution',str(row['id']),document))
                eligible.append(row)
        if not eligible:
            return None
        try:
            vectors = self._vectors(documents,model)
            if vectors is None:
                return None
            matches = []
            for row,vector in zip(eligible,vectors[1:]):
                similarity = cosine(vectors[0],vector)
                score = self.rank(row,similarity)
                if score>=threshold:
                    matches.append(dict(row,confidence=score,semantic_similarity=similarity,
                        reason=f"semantic match={similarity:.2f}; {row['verification_status']}, "
                               f"explicit teaching={row['source']=='user'}, successes={row['success_count']}; score={score:.2f}"))
            return max(matches,key=lambda row:(row['confidence'],row['source']=='user',row['updated_at'])) if matches else None
        except (sqlite3.Error,OSError,ValueError,TypeError,KeyError,OverflowError,struct.error):
            return None
