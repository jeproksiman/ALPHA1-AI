"""One companion context over AlphaMemory, shared by every speech/text provider."""
import re
from core.brain.intent_engine import normalize
from core.learning.knowledge_extractor import contains_secret
from memory.config_manager import load_api_keys


class Companion:
    def __init__(self, memory, identity=None):
        self.memory = memory
        cfg = identity if identity is not None else load_api_keys()
        self.identity = {k:str(cfg.get(field) or default)[:100] for k,field,default in
                         [('assistant_name','assistant_name','JARVIS'),('preferred_name','user_name','')]}
        self.identity = {k:v for k,v in self.identity.items() if not contains_secret(v)}
        self.profile_reader = lambda: {}

    def profile_context(self):
        lines=[]
        profile=self.profile_reader()
        for category in ('identity','preferences','projects','goals'):
            entries=profile.get(category,{})
            if not isinstance(entries,dict):
                continue
            for key,entry in entries.items():
                value=entry.get('value') if isinstance(entry,dict) else entry
                if isinstance(value,str) and not contains_secret(str(key)+' '+value):
                    lines.append(str(key)[:60]+': '+value[:120])
        return '\n'.join(lines)[:500]

    def facts(self):
        return dict(self.identity, **self.memory.companion_facts())

    def handle(self, text):
        query = normalize(text)
        state = self.memory.conversation_state()
        if query in {'what were we just talking about','what were we talking about','what were we just discussing',
                     'what were we testing','what were we discussing'}:
            return 'We were discussing: '+state['topic'] if state.get('topic') else 'There is no saved conversation topic yet.'
        if query in {'what am i working on','what are we working on'}:
            project = self.facts().get('active_project') or state.get('project') or state.get('topic')
            return 'You are working on: '+project if project else 'You have not told me what you are working on yet.'
        if query in {'what did i tell you earlier','remember this'}:
            rows = self.memory.recent_conversation(1)
            if query == 'remember this' and rows:
                self.memory.set_companion_fact('note', rows[-1]['user_input'][:500])
                return 'I saved your previous message locally.'
            return 'You told me: '+rows[-1]['user_input'] if rows else 'There is no earlier saved message yet.'
        if query == 'forget that':
            self.memory.forget_last_conversation()
            return 'I forgot the latest conversation and its saved note/goal.'
        if query == 'what do you remember about me':
            lines=[k.replace('_',' ').title()+': '+v for k,v in self.facts().items() if v]
            if self.profile_context():
                lines.append(self.profile_context())
            return '\n'.join(lines) or 'You have not shared any personal facts yet.'
        patterns = [(r'(?:my name is|call me) (.+)', 'preferred_name'),
                    (r'(?:your name is|call yourself) (.+)', 'assistant_name'),
                    (r'i prefer (.+)', 'preferences'), (r'(?:i am|i\x27m) working on (.+)', 'active_project'),
                    (r'remember(?: that| this)?[ :]+(.+)', 'note')]
        for pattern, name in patterns:
            match = re.fullmatch(pattern,text.strip(),re.I)
            if match:
                value = match[1].strip()
                if re.search(r'\b(next|goal|want to)\b',value,re.I) and name == 'note':
                    name = 'goal'
                if self.memory.set_companion_fact(name,value):
                    return 'I saved that locally.'
                return 'That information cannot be stored safely.'
        return None

    def record(self, user, answer, mode='offline', inspection=False, project=''):
        if contains_secret(user+'\n'+answer):
            return False
        if mode=='live' and user:
            if normalize(user)=='forget that':
                self.handle(user)
                return True
            if re.match(r'^(my name is|call me|your name is|call yourself|i prefer|i am working on|remember that)\b',user,re.I):
                self.handle(user)
        # Questions/inspection replies do not displace the topic being recalled.
        topic = '' if inspection or re.match(r'^(what|why|how|who|where|when|can you)\b',user,re.I) else user[:300]
        next_step = user[:300] if re.search(r'\b(next|goal|want to)\b',user,re.I) else ''
        known_project = re.search(r"\b(?:we are|i am|we're|i'm) working on (.+)",user,re.I)
        if known_project:
            project=known_project[1][:300]
            self.memory.set_companion_fact('active_project',project)
        return self.memory.conversation(user[:4000],answer[:4000],mode,topic,project,next_step,
                                        .9 if next_step or user.lower().startswith('remember') else .1 if inspection else .5)

    def prompt(self, current=''):
        state = self.memory.conversation_state()
        lines = ['[Shared companion identity and recent local context]',
                 'Same companion across providers. Be concise, helpful and honest; never invent personal facts.']
        lines += [k.replace('_',' ').title()+': '+v[:120] for k,v in self.facts().items() if v]
        if state.get('summary'):
            lines.append(state['summary'][:600])
        if self.profile_context():
            lines.append(self.profile_context())
        for row in self.memory.recent_conversation(4):
            lines.append('User: '+row['user_input'][:220])
            lines.append('Assistant: '+row['response'][:180])
        return '\n'.join(line for line in lines if not contains_secret(line))[:1800]
