from unittest.mock import Mock, patch
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from core.providers.ollama_provider import OllamaProvider
from core.offline_brain import OfflineBrain
from core.memory.semantic_index import SemanticIndex, cosine
from core.learning.knowledge_extractor import KnowledgeExtractor
from memory.alpha_memory import AlphaMemory
from memory.config_manager import get_offline_brain_settings


@pytest.fixture
def memory(tmp_path):
    memory=AlphaMemory(tmp_path/'brain.db')
    yield memory
    memory.close()


@pytest.fixture
def teacher():
    provider=Mock()
    provider.embedding_model='nomic-embed-text:latest'
    provider.embed.side_effect=lambda texts:{'ok':True,'embeddings':[[1.,0.] for text in texts]}
    provider.generate.return_value={'ok':True,'text':'A concise primary-model answer.'}
    return provider


def http_provider():
    session=Mock()
    session.get.return_value.status_code=200
    session.get.return_value.json.return_value={'models':[{'name':name} for name in
        ('gemma4:e4b','qwen3.5:0.8b','nomic-embed-text:latest')]}
    session.post.return_value.status_code=200
    session.post.return_value.json.return_value={'done':True,'message':{'content':'A concise local answer.'},'embeddings':[[1.,0.]]}
    return OllamaProvider(model='gemma4:e4b',session=session),session


def test_role_defaults_and_env_overrides():
    with patch('memory.config_manager.load_api_keys',return_value={}),patch.dict('os.environ',{},clear=True):
        config=get_offline_brain_settings()
        assert config['ollama_model']=='gemma4:e4b'
        assert config['ollama_fast_model']=='qwen3.5:0.8b'
        assert config['embedding_model']=='nomic-embed-text:latest'
        assert config['semantic_enabled']
        assert not config['auto_learn']
    with patch('memory.config_manager.load_api_keys',return_value={}),patch.dict('os.environ',{'ALPHA_SEMANTIC_MEMORY':'false','ALPHA_OLLAMA_FAST_MODEL':'other'},clear=True):
        config=get_offline_brain_settings()
        assert not config['semantic_enabled'] and config['ollama_fast_model']=='other'


def test_local_config_role_keys_preserved():
    with patch('memory.config_manager.load_api_keys',return_value={'alpha_ollama_model':'custom:local'}),patch.dict('os.environ',{},clear=True):
        assert get_offline_brain_settings()['ollama_model']=='custom:local'


def test_primary_fast_embedding_role_separation():
    provider,session=http_provider()
    assert provider.generate('A difficult question')['ok']
    assert session.post.call_args.kwargs['json']['model']=='gemma4:e4b'
    assert provider.lightweight('Short passage',task='summarize')['ok']
    assert session.post.call_args.kwargs['json']['model']=='qwen3.5:0.8b'
    assert provider.embed('Memory text')['ok']
    assert session.post.call_args.kwargs['json']['model']=='nomic-embed-text:latest'
    assert session.post.call_args.args[0].endswith('/api/embed')


def test_fast_only_allows_lightweight_explicit_tasks():
    provider,session=http_provider()
    assert not provider.lightweight('Hard reasoning',task='reason')['ok']
    assert not provider.lightweight('x'*2001)['ok']
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_missing_fast_does_not_switch_roles():
    provider,session=http_provider()
    provider.fast_model='absent'
    assert not provider.lightweight('Short text')['ok']
    session.post.assert_not_called()
    assert provider.generate('Unknown reasoning')['ok']
    assert session.post.call_args.kwargs['json']['model']=='gemma4:e4b'


@pytest.mark.parametrize('model',['gemma4:cloud','org/model-cloud','https://host/model'])
def test_cloud_model_rejected_for_all_roles(model):
    provider,session=http_provider()
    session.get.return_value.json.return_value={'models':[{'name':model}]}
    provider.model=provider.fast_model=provider.embedding_model=model
    assert not provider.generate('Unknown')['ok']
    assert not provider.lightweight('Short')['ok']
    assert not provider.embed('Memory')['ok']
    session.post.assert_not_called()


def test_remote_metadata_and_malformed_model_list():
    provider,session=http_provider()
    session.get.return_value.json.return_value={'models':[{'name':'gemma4:e4b','remote_host':'https://host'}]}
    assert not provider.generate('Unknown')['ok']
    session.post.assert_not_called()
    for data in ({}, {'models':[None]}, {'models':'invalid'}):
        session.get.return_value.json.return_value=data
        assert not provider.status()['available']


@pytest.mark.parametrize('url',['https://127.0.0.1:11434','http://example.com','http://127.0.0.1:11434/api','http://user:password@localhost:11434',
                              'http://[bad','http://127.0.0.1:invalid'])
def test_nonlocal_urls_never_send_http(url):
    provider,session=http_provider()
    other=OllamaProvider(url=url,model='gemma4:e4b',session=session)
    assert not other.generate('question')['ok']
    assert not other.embed('text')['ok']
    session.get.assert_not_called()
    session.post.assert_not_called()


@pytest.mark.parametrize('vectors',[[[float('nan')]],[[0.,0.]],[[True]],[],[[1.,0.],[1.,0.]]])
def test_embedding_validation(vectors):
    provider,session=http_provider()
    session.post.return_value.json.return_value={'embeddings':vectors}
    assert not provider.embed('Text')['ok']


def test_semantic_is_lazy_and_skips_empty_store(memory,teacher):
    index=SemanticIndex(memory,teacher)
    assert not index.cache_path.exists()
    assert index.search('a question') is None
    assert not index.cache_path.exists()
    teacher.embed.assert_not_called()


def test_semantic_before_reasoning_and_cache_has_no_raw_text(memory,teacher):
    item=memory.store_solution('Relational persistence','SQLite retains data on disk.',source='user',confidence=.9)
    brain=OfflineBrain(memory,teacher,settings={})
    result=brain.process('Where are the records kept?')
    assert result['intent']=='semantic_solution' and result['knowledge_id']==item
    teacher.generate.assert_not_called()
    calls=teacher.embed.call_count
    assert brain.process('Where are the records kept?')['source']=='memory'
    assert teacher.embed.call_count==calls
    with sqlite3.connect(brain.semantic.cache_path) as db:
        names={row[1] for row in db.execute('PRAGMA table_info(embeddings)')}
        assert 'text' not in names and 'response' not in names
        assert db.execute('SELECT typeof(vector) FROM embeddings LIMIT 1').fetchone()[0]=='blob'


def test_strong_local_routes_never_call_any_model(memory,teacher):
    brain=OfflineBrain(memory,teacher,settings={})
    brain.register_command('known command',lambda ctx:'Done')
    memory.store_solution('database storage','A verified reusable answer.',source='user',confidence=.9)
    brain.process('known command')
    brain.process('database storage')
    brain.process('database storages')
    teacher.embed.assert_not_called()
    teacher.generate.assert_not_called()
    teacher.lightweight.assert_not_called()


def test_alias_and_routine_skip_embeddings(memory,teacher):
    brain=OfflineBrain(memory,teacher,settings={})
    brain.register_command('open app',lambda ctx:{'text':'Opened app','success':True},action_name='open_app',
        action_args={'app_name':'app'},repeatable=True,routine_allowed=True)
    memory.add_alias('editor','open app')
    brain.process('editor')
    brain.routines.save('mode',['open app'])
    brain.process('mode')
    teacher.embed.assert_not_called()
    teacher.generate.assert_not_called()


@pytest.mark.parametrize('settings',[{'semantic_enabled':False},{'embedding_provider':'cloud'}])
def test_disabled_semantic_uses_primary(memory,teacher,settings):
    memory.store_solution('Persistence','An answer about local databases.',source='user',confidence=.9)
    brain=OfflineBrain(memory,teacher,settings=settings)
    assert brain.process('A different question')['source']=='ollama'
    teacher.embed.assert_not_called()
    teacher.generate.assert_called_once()


def test_embedding_error_graceful_reasoning_fallback(memory,teacher):
    memory.store_solution('Persistence','An answer about local databases.',source='user',confidence=.9)
    teacher.embed.return_value={'ok':False,'embeddings':[]}
    teacher.embed.side_effect=None
    brain=OfflineBrain(memory,teacher,settings={})
    assert brain.process('A different question')['source']=='ollama'


def test_candidate_never_semantically_answers(memory,teacher):
    memory.store_solution('Persistence','A candidate answer.',confidence=.7)
    brain=OfflineBrain(memory,teacher,settings={})
    assert brain.process('An unfamiliar question')['source']=='ollama'
    teacher.embed.assert_not_called()


def test_semantic_correction_outweighs_older_equal_match(memory,teacher):
    old=memory.store_solution('Older question','An older verified solution.',source='verified_teacher',confidence=.9)
    new=memory.store_solution('Corrected question','The original response to replace.',confidence=.6)
    memory.replace_solution(new,'The explicitly corrected reusable answer.')
    result=SemanticIndex(memory,teacher).search('Unfamiliar related request')
    assert result['id']==new and result['id']!=old


def test_correction_invalidates_cached_document(memory,teacher):
    item=memory.store_solution('Database','The original verified answer.',source='user',confidence=.9)
    index=SemanticIndex(memory,teacher)
    index.search('New request')
    teacher.embed.reset_mock()
    memory.replace_solution(item,'The corrected verified answer.')
    index.search('New request')
    teacher.embed.assert_called_once()
    assert len(teacher.embed.call_args.args[0])==1
    assert 'corrected' in teacher.embed.call_args.args[0][0]


def test_semantic_disagreement_and_cosine(memory,teacher):
    memory.store_solution('Database','A verified database answer.',source='user',confidence=.9)
    teacher.embed.side_effect=lambda texts:{'ok':True,'embeddings':[[1.,0.] if i==0 else [0.,1.] for i,_ in enumerate(texts)]}
    assert SemanticIndex(memory,teacher).search('Unrelated request') is None
    assert cosine([1,0],[0,1])==0
    assert cosine([1],[1,0])==0
    assert cosine([0,0],[0,0])==0


def test_secret_inputs_and_legacy_rows_never_embedded(memory,teacher):
    item=memory.store_solution('Database','A safe-looking answer.',source='user',confidence=.9)
    with memory.db:
        memory.db.execute('UPDATE learned_knowledge SET response=? WHERE id=?',('token=synthetic-secret',item))
    index=SemanticIndex(memory,teacher)
    assert index.search('token=synthetic-secret') is None
    assert index.search('unfamiliar question') is None
    teacher.embed.assert_not_called()
    assert not index.cache_path.exists()


def test_distillation_does_not_store_raw_dump(memory,teacher):
    raw='A reusable opening sentence. '+('Further detail which does not need long-term storage. '*25)
    teacher.generate.return_value={'ok':True,'text':raw}
    brain=OfflineBrain(memory,teacher,settings={})
    brain.process('How do I retain this knowledge?')
    assert memory.counts()['knowledge']==0
    result=brain.process('that worked')
    saved=memory.get_knowledge(result['knowledge_id'])
    assert len(saved['response'])<=800 and saved['response']!=raw
    assert saved['verification_status']=='verified'
    assert KnowledgeExtractor().extract('question','x'*3000) is None


def test_startup_reports_roles_without_embedding_request(memory):
    provider,session=http_provider()
    brain=OfflineBrain(memory,provider,settings={})
    lines=brain.startup_report()
    assert '[OLLAMA] gemma4:e4b ready' in lines
    assert '[OLLAMA FAST] qwen3.5:0.8b ready' in lines
    assert '[SEMANTIC MEMORY] nomic-embed-text:latest ready' in lines
    session.get.assert_called_once()
    session.post.assert_not_called()


def test_known_command_does_not_wait_for_reasoning_in_flight(memory,teacher):
    entered,release=Event(),Event()
    def generate(text,context):
        entered.set()
        release.wait(5)
        return {'ok':True,'text':'An unknown request answer.'}
    teacher.generate.side_effect=generate
    brain=OfflineBrain(memory,teacher,settings={})
    brain.register_command('known command',lambda ctx:'Immediate local answer')
    with ThreadPoolExecutor(max_workers=2) as pool:
        unknown=pool.submit(brain.process,'unknown request')
        assert entered.wait(2)
        try:
            known=pool.submit(brain.process,'known command')
            assert known.result(timeout=2)['text']=='Immediate local answer'
        finally:
            release.set()
        assert unknown.result(timeout=2)['source']=='ollama'


def test_semantic_snapshot_revalidated_after_feedback(memory,teacher):
    item=memory.store_solution('Database','An old verified answer.',source='user',confidence=.9)
    index=Mock()
    def search(*args,**kwargs):
        snapshot=memory.get_knowledge(item)
        memory.record_feedback(item,False,reject=True)
        return dict(snapshot,reason='Old semantic snapshot')
    index.search.side_effect=search
    brain=OfflineBrain(memory,teacher,settings={},semantic_index=index)
    assert brain.process('An unfamiliar query')['source']=='ollama'
