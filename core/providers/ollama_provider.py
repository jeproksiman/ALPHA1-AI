"""Bounded, text-only Ollama fallback. Never starts services or pulls models."""
from urllib.parse import urlsplit
import requests


class OllamaProvider:
    def __init__(self, url='http://127.0.0.1:11434', model='', enabled=True, timeout=45, session=None):
        self.url = str(url).rstrip('/')
        self.model = str(model or '').strip()
        self.enabled = enabled
        self.timeout = max(1, min(float(timeout), 120))
        self.session = session or requests.Session()
        self.session.trust_env = False  # Local credentials/prompts must not traverse a proxy.
        parsed = urlsplit(self.url)
        self.valid_url = (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
                          and not parsed.username and not parsed.password and parsed.path in ('', '/')
                          and not parsed.query and not parsed.fragment)

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
            models = [item['name'] for item in data['models'] if isinstance(item.get('name'), str)]
            return {'available': True, 'models': models, 'error': None}
        except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError):
            return {'available': False, 'models': [], 'error': 'Ollama is unavailable. Start Ollama locally to use fallback.'}

    def is_available(self):
        return self.status()['available']

    def chat(self, messages):
        status = self.status()
        if not status['available']:
            return {'ok': False, 'text': status['error']}
        model = self.model or next(iter(sorted(status['models'])), '')
        if not model:
            return {'ok': False, 'text': 'No installed Ollama model. Configure ALPHA_OLLAMA_MODEL after installing a model.'}
        if model not in status['models'] and model + ':latest' not in status['models']:
            return {'ok': False, 'text': 'Configured Ollama model is not installed. Set ALPHA_OLLAMA_MODEL to an installed model.'}
        try:
            response = self.session.post(self.url + '/api/chat',
                json={'model': model, 'messages': messages, 'stream': False, 'options': {'num_predict': 600}},
                timeout=(3, self.timeout), allow_redirects=False)
            if response.status_code != 200:
                raise ValueError('Invalid HTTP status')
            data = response.json()
            text = data['message']['content']
            if not isinstance(text, str) or not text.strip() or data.get('done') is not True:
                raise ValueError('Incomplete response')
            return {'ok': True, 'text': text.strip()}
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return {'ok': False, 'text': 'Ollama could not complete the request. Check the local service and model.'}

    def generate(self, text, context=None):
        messages = [{'role': 'system', 'content':
            'You are ALPHA, a local assistant. Give a concise text answer. '
            'You cannot execute tools or claim to have performed actions.'}]
        # Only explicit text context is sent, never arbitrary application state/config.
        if context and isinstance(context.get('summary'), str):
            messages.append({'role': 'system', 'content': context['summary'][:2000]})
        messages.append({'role': 'user', 'content': text})
        return self.chat(messages)
