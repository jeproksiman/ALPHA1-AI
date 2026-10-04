"""Bounded, text-only Ollama fallback. Never starts services or pulls models."""
from urllib.parse import urlsplit
import math
import re
import requests
from core.learning.knowledge_extractor import contains_secret


class OllamaProvider:
    def __init__(self, url='http://127.0.0.1:11434', model='', enabled=True, timeout=45, session=None,
                 fast_model='qwen3.5:0.8b', embedding_model='nomic-embed-text:latest'):
        self.url = str(url).rstrip('/')
        self.model = str(model or '').strip()
        self.enabled = enabled
        self.fast_model = str(fast_model or '').strip()
        self.embedding_model = str(embedding_model or '').strip()
        self.timeout = max(1, min(float(timeout), 120))
        self.session = session or requests.Session()
        self.session.trust_env = False  # Local credentials/prompts must not traverse a proxy.
        try:
            parsed = urlsplit(self.url)
            self.valid_url = (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
                              and not parsed.username and not parsed.password and parsed.path in ('', '/')
                              and not parsed.query and not parsed.fragment)
            if parsed.port is not None and not 1<=parsed.port<=65535:
                self.valid_url = False
        except ValueError:
            self.valid_url = False

    def status(self):
        if not self.enabled:
            return {'available': False, 'models': [], 'error': 'Ollama fallback is disabled.'}
        if not self.valid_url:
            return {'available': False, 'models': [], 'error': 'Configure a loopback HTTP Ollama URL.'}
        try:
            response = self.session.get(self.url + '/api/tags', timeout=(2, 3), allow_redirects=False)
            if response.status_code != 200:
                raise ValueError('Invalid HTTP status')
            data = response.json()
            models = []
            blocked = []
            for item in data['models']:
                name = item.get('name')
                if not isinstance(name,str):
                    continue
                if not self.local_model(name) or item.get('remote_model') or item.get('remote_host'):
                    blocked.append(name)
                else:
                    models.append(name)
            return {'available': True, 'models': models, 'blocked_models':blocked, 'error': None}
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError):
            return {'available': False, 'models': [], 'error': 'Ollama is unavailable. Start Ollama locally to use fallback.'}

    def is_available(self):
        return self.status()['available']

    @staticmethod
    def local_model(model):
        return bool(model and re.fullmatch(r'[\w./:-]{1,100}',model) and 'cloud' not in model.casefold()
                    and '://' not in model and not contains_secret(model))

    def _model(self, role, status):
        configured = {'primary':self.model,'fast':self.fast_model,'embedding':self.embedding_model}[role]
        if not configured and role=='primary':
            configured = next((name for name in sorted(status['models']) if 'embed' not in name.lower()),'')
        if not self.local_model(configured):
            return None
        return configured if configured in status['models'] or configured+':latest' in status['models'] else None

    def model_status(self):
        status = self.status()
        return {'available':status['available'],'installed':status['models'],
                'roles':{role:{'model':model,'ready':bool(status['available'] and self._model(role,status))}
                         for role,model in [('primary',self.model),('fast',self.fast_model),('embedding',self.embedding_model)]}}

    def chat(self, messages, role='primary', max_tokens=600, think=None):
        if role not in ('primary','fast'):
            return {'ok':False,'text':'Invalid generation role.'}
        if contains_secret(str(messages)):
            return {'ok':False,'text':'Sensitive prompts are not sent to the model.'}
        status = self.status()
        if not status['available']:
            return {'ok': False, 'text': status['error']}
        model = self._model(role,status)
        if not model:
            available = ', '.join(name for name in status['models'] if len(name)<=100) or 'none'
            return {'ok': False, 'text': 'Configured Ollama model is not installed. Available models: '
                    + available + '. Configure an installed local model explicitly.'}
        try:
            payload = {'model':model,'messages':messages,'stream':False,
                       'options':{'num_predict':max(1,min(max_tokens,600))}}
            if think is not None:
                payload['think'] = bool(think)
            response = self.session.post(self.url + '/api/chat',
                json=payload,
                timeout=(3, self.timeout), allow_redirects=False)
            if response.status_code != 200:
                raise ValueError('Invalid HTTP status')
            data = response.json()
            text = data['message']['content']
            if not isinstance(text, str) or not text.strip() or data.get('done') is not True:
                raise ValueError('Incomplete response')
            if contains_secret(text):
                raise ValueError('Sensitive response')
            return {'ok': True, 'text': text.strip(), 'model':model}
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return {'ok': False, 'text': 'Ollama could not complete the request. Check the local service and model.'}

    def generate(self, text, context=None):
        messages = [{'role': 'system', 'content':
            'You are ALPHA\'s local teacher and reasoning fallback. ALPHA already '
            'handles registered local actions and verified knowledge. Answer only the '
            'unresolved question with a concise, reusable text solution. Do not claim '
            'that actions were executed. You cannot execute tools. Do not request or '
            'repeat passwords, tokens, private keys, or authentication cookies.'}]
        # Only explicit text context is sent, never arbitrary application state/config.
        if context and isinstance(context.get('summary'), str):
            messages.append({'role': 'system', 'content': context['summary'][:2000]})
        messages.append({'role': 'user', 'content': text})
        return self.chat(messages)

    def lightweight(self, text, task='summarize', options=None):
        """Explicit opt-in only. Output is advisory text, never an action authorization."""
        if task not in ('classify','summarize','tags','choose') or not isinstance(text,str) or len(text)>2000:
            return {'ok':False,'text':'Use the primary model for this request.'}
        if options and (len(options)>10 or any(not isinstance(item,str) or len(item)>100 for item in options)):
            return {'ok':False,'text':'Too many classification options.'}
        prompt = f'Perform only this lightweight task: {task}. Give a short text result; do not execute actions.'
        if options:
            prompt += ' Choose only from: '+', '.join(options)
        return self.chat([{'role':'system','content':prompt},{'role':'user','content':text}],role='fast',max_tokens=120,think=False)

    def embed(self, texts):
        if isinstance(texts,str):
            texts = [texts]
        if not isinstance(texts,list) or not 1<=len(texts)<=32 or any(not isinstance(t,str) or not t.strip() or len(t)>3000 or contains_secret(t) for t in texts):
            return {'ok':False,'embeddings':[],'error':'Embedding input is invalid or sensitive.'}
        status = self.status()
        model = self._model('embedding',status) if status['available'] else None
        if not model:
            return {'ok':False,'embeddings':[],'error':'Configured local embedding model is unavailable.'}
        try:
            response = self.session.post(self.url+'/api/embed',json={'model':model,'input':texts,'truncate':False},
                timeout=(3,min(self.timeout,45)),allow_redirects=False)
            if response.status_code!=200:
                raise ValueError('Invalid HTTP status')
            vectors = response.json()['embeddings']
            if len(vectors)!=len(texts) or not vectors or not 1<=len(vectors[0])<=8192:
                raise ValueError('Invalid vector count')
            dimension = len(vectors[0])
            for vector in vectors:
                if len(vector)!=dimension or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vector) or not any(vector):
                    raise ValueError('Invalid vector')
            return {'ok':True,'embeddings':vectors,'model':model}
        except (requests.RequestException,ValueError,TypeError,KeyError,IndexError):
            return {'ok':False,'embeddings':[],'error':'Local embedding request failed.'}
