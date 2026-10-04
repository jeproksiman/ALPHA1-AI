import json
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core.offline_brain import OfflineBrain
from core.brain.app_commands import configure_app_commands
from core.context.context_manager import ContextManager
from core.learning.consolidator import MemoryConsolidator
from core.routines.pattern_detector import PatternDetector
from memory.alpha_memory import AlphaMemory
from core.action_loader import _call_handler
from actions.local_workspace import local_workspace


@pytest.fixture
def memory(tmp_path):
    store = AlphaMemory(tmp_path/'brain.db')
    yield store
    store.close()


@pytest.fixture
def brain(memory):
    provider = Mock()
    provider.generate.return_value = {'ok':True,'text':'Run python -m pip install PyQt6 to install the missing package.'}
    return OfflineBrain(memory,provider,settings={})


@pytest.fixture
def registry(brain):
    registry = Mock()
    registry.has.side_effect = lambda name:name in ('open_app','local_workspace')
    def run(name,args,ctx):
        if name=='open_app':
            return 'Opened '+args['app_name']+'.'
        if args['operation']=='open_project':
            return 'Opened project directory.'
        if args['operation']=='run_project':
            return 'Started project main.py.'
        return 'Sent close request to '+args['target']+'.'
    registry.run.side_effect = run
    configure_app_commands(brain,registry,['vscode','chrome','spotify','powershell'])
    return registry


def approve_pending(brain):
    gate = brain.confirmer
    callback = gate.call_args.args[2]
    return callback()


def test_user_success_verifies_and_reuses_without_ollama(brain):
    brain.process('My Python says ModuleNotFoundError: PyQt6')
    feedback = brain.process('that worked')
    knowledge = brain.memory.get_knowledge(feedback['knowledge_id'])
    assert knowledge['verification_status']=='verified'
    assert knowledge['confidence']==0.85
    assert knowledge['success_count']==1
    brain.provider.reset_mock()
    result = brain.process('My Python says ModuleNotFoundError: PyQt6')
    assert result['source']=='memory'
    assert 'verified' in result['reason']
    assert brain.memory.get_knowledge(knowledge['id'])['use_count']==1
    brain.provider.generate.assert_not_called()


def test_semantically_similar_missing_module(brain):
    brain.process('My Python says ModuleNotFoundError: PyQt6')
    brain.process('that worked')
    brain.provider.reset_mock()
    assert brain.process('PyQt6 module missing')['source']=='memory'
    brain.provider.generate.assert_not_called()
    assert brain.process('PyQt5 module missing')['source']=='ollama'


def test_duplicate_feedback_cannot_inflate_confidence(brain):
    brain.process('missing PyQt6')
    first = brain.process('that worked')
    brain.process('perfect')
    state = brain.memory.get_knowledge(first['knowledge_id'])
    assert state['success_count']==1
    assert state['confidence']==0.85
    assert brain.memory.db.execute('SELECT count(*) FROM confidence_events').fetchone()[0]==1


def test_feedback_tracks_latest_meaningful_interaction(brain,registry):
    brain.process('missing PyQt6')
    brain.process('open chrome')
    assert brain.process('that worked')['text']=='Action feedback recorded.'
    assert brain.memory.counts()['knowledge']==0
    assert brain.memory.db.execute('SELECT count(*) FROM interaction_history WHERE intent != ?',("conversation",)).fetchone()[0]==1


def test_pending_action_cannot_verify_previous_answer(brain,registry):
    brain.process('missing PyQt6')
    brain.process('open vscode')
    assert 'Which recent' in brain.process('that worked')['text']
    assert brain.memory.counts()['knowledge']==0


def test_failed_fallback_is_feedback_barrier(brain):
    brain.process('missing PyQt6')
    brain.provider.generate.return_value={'ok':False,'text':'Offline'}
    brain.process('another question')
    assert 'Which recent' in brain.process('that worked')['text']
    assert brain.memory.counts()['knowledge']==0


def test_negative_feedback_demotes_then_rejects(brain):
    brain.process('missing PyQt6')
    item = brain.process('that worked')['knowledge_id']
    brain.process('that was wrong')
    state = brain.memory.get_knowledge(item)
    assert state['confidence']==pytest.approx(0.65)
    assert state['verification_status']=='candidate'
    brain.process('missing PyQt6')
    brain.process('it failed')
    assert brain.memory.get_knowledge(item)['verification_status']=='rejected'
    assert brain.memory.get_knowledge(item)['failure_count']==2
    assert brain.memory.search('missing PyQt6',eligible_only=True) is None


def test_first_failure_retained_and_explicit_veto_disables(brain):
    brain.process('missing PyQt6')
    item=brain.process('wrong')['knowledge_id']
    assert brain.memory.get_knowledge(item)['failure_count']==1
    assert brain.memory.get_knowledge(item)['confidence']==pytest.approx(0.4)
    brain.process('missing PyQt6')
    brain.process("don't use that solution")
    assert brain.memory.get_knowledge(item)['verification_status']=='rejected'


def test_repeated_success_becomes_trusted(brain):
    brain.process('missing PyQt6')
    item = brain.process('it worked')['knowledge_id']
    for _ in range(2):
        brain.process('missing PyQt6')
        brain.process('perfect')
    state = brain.memory.get_knowledge(item)
    assert state['verification_status']=='trusted'
    assert state['confidence']<=1


def test_remember_is_candidate_not_verification(brain):
    brain.process('missing PyQt6')
    item = brain.process('remember this')['knowledge_id']
    assert brain.memory.get_knowledge(item)['verification_status']=='candidate'
    assert brain.memory.get_knowledge(item)['confidence']==0.60
    assert brain.process('missing PyQt6')['source']=='ollama'


def test_correction_preserves_original_and_restores_rejected(brain):
    brain.process('missing PyQt6')
    item = brain.process('remember this')['knowledge_id']
    brain.memory.record_feedback(item,False)
    brain.memory.record_feedback(item,False)
    brain.process('the correct solution is Use the Python environment where PyQt6 is installed.')
    state = brain.memory.get_knowledge(item)
    assert state['verification_status']=='verified'
    assert state['source']=='user'
    correction = brain.memory.db.execute('SELECT * FROM corrections').fetchone()
    assert correction['wrong_result'].startswith('Run python')
    assert correction['corrected_result'].startswith('Use the Python')


def test_conflicting_candidate_cannot_verify_old_response(brain):
    brain.process('missing PyQt6')
    item = brain.process('remember this')['knowledge_id']
    brain.provider.generate.return_value={'ok':True,'text':'A different proposed fix which conflicts with the earlier answer.'}
    brain.process('missing PyQt6')
    assert 'conflicts' in brain.process('that worked')['text']
    assert brain.memory.get_knowledge(item)['verification_status']=='candidate'
    assert brain.memory.db.execute('SELECT count(*) FROM knowledge_conflicts').fetchone()[0]==1


def test_automatic_feedback_deltas_and_event_receipts(memory):
    item = memory.store_solution('topic','A reusable answer.',source='user',confidence=0.9)
    assert memory.record_feedback(item,True,automatic=True,event_key='event')
    assert memory.get_knowledge(item)['confidence']==pytest.approx(0.95)
    assert not memory.record_feedback(item,True,automatic=True,event_key='event')
    memory.record_feedback(item,False,automatic=True)
    assert memory.get_knowledge(item)['confidence']==pytest.approx(0.85)
    events = memory.inspect('events')
    assert events[0]['delta']==pytest.approx(-0.1)


def test_alias_learning_update_and_persistence(brain,registry,tmp_path):
    assert brain.process('When I say editor, I mean VS Code.')['intent']=='alias_learning'
    assert brain.memory.get_alias('editor')['target']=='open vscode'
    brain.confirmer=Mock(return_value='pending')
    brain.process('open editor')
    approve_pending(brain)
    assert registry.run.call_args.args[1]=={'app_name':'vscode'}
    brain.process('When I say editor, I mean Chrome.')
    other=AlphaMemory(brain.memory.db.execute('PRAGMA database_list').fetchone()[2])
    assert other.get_alias('editor')['target']=='open chrome'
    other.close()
    brain.process('open editor')
    approve_pending(brain)
    assert registry.run.call_args.args[1]=={'app_name':'chrome'}
    brain.provider.generate.assert_not_called()


def test_correction_alias_and_action_phrase(brain,registry):
    brain.process('no, by browser I mean Chrome')
    brain.process('When I say music, open Spotify')
    assert brain.memory.get_alias('browser')['target']=='open chrome'
    assert brain.memory.get_alias('music')['target']=='open spotify'
    assert brain.process('open browser')['confirmation_required']


def test_alias_cycle_fails_closed(brain):
    brain.memory.add_alias('one','two')
    brain.memory.add_alias('two','one')
    assert brain.adaptive.resolve_command('one') is None
    assert brain.process('one')['source']=='ollama'


def test_context_it_targets_last_successful_app(brain,registry):
    brain.confirmer=Mock(return_value='pending')
    brain.process('open spotify')
    result=brain.process('close it')
    assert result['confirmation_required']
    assert registry.run.call_count==1
    approve_pending(brain)
    assert registry.run.call_args.args[1]=={'operation':'close_app','target':'spotify'}
    assert registry.run.call_args.args[2]['alpha_confirmed'] is True
    brain.provider.generate.assert_not_called()


def test_ambiguous_and_stale_context_do_not_execute(brain,registry):
    assert brain.process('close it')['intent']=='clarification'
    registry.run.assert_not_called()
    brain.process('open spotify')
    brain.process('a different question')
    registry.run.reset_mock()
    assert brain.process('close it')['intent']=='clarification'
    registry.run.assert_not_called()
    brain.context.history.clear()
    brain.process('open spotify')
    brain.context.history[-1].at-=10000
    registry.run.reset_mock()
    assert brain.process('close it')['intent']=='clarification'
    registry.run.assert_not_called()


def test_context_is_bounded_and_rejects_sensitive_values():
    context=ContextManager(limit=3)
    for i in range(10):
        context.remember(str(i),{'text':'answer','intent':'teacher_answer'})
    assert len(context.history)==3
    assert context.remember('cookie=synthetic',{'text':'answer'}) is None


def test_project_alias_open_and_run_require_gate(brain,registry,tmp_path):
    path=tmp_path/'project'
    path.mkdir()
    (path/'main.py').write_text('print("hello")')
    brain.process('When I say Alpha project, use '+str(path))
    brain.confirmer=Mock(return_value='pending')
    brain.process('open my Alpha project')
    approve_pending(brain)
    assert registry.run.call_args.args[1]['operation']=='open_project'
    registry.run.reset_mock()
    assert brain.process('run it')['confirmation_required']
    registry.run.assert_not_called()
    approve_pending(brain)
    assert registry.run.call_args.args[1]['operation']=='run_project'
    assert registry.run.call_args.args[2]['alpha_confirmed'] is True


def test_routine_create_persist_order_and_execute(brain,registry):
    result=brain.process('When I say coding mode: open VS Code, open Chrome, open Spotify.')
    assert 'saved' in result['text']
    routine=brain.routines.get('coding mode')
    assert [json.loads(step['action_args'])['parameters']['app_name'] for step in routine['steps']]==['vscode','chrome','spotify']
    brain.confirmer=Mock(return_value='pending')
    assert brain.process('coding mode')['confirmation_required']
    registry.run.assert_not_called()
    approve_pending(brain)
    assert [call.args[1]['app_name'] for call in registry.run.call_args_list]==['vscode','chrome','spotify']
    assert brain.routines.get('coding mode')['success_count']==1
    other=AlphaMemory(brain.memory.db.execute('PRAGMA database_list').fetchone()[2])
    second=OfflineBrain(other,Mock(),settings={})
    configure_app_commands(second,registry,['vscode','chrome','spotify','powershell'])
    assert len(second.routines.get('coding mode')['steps'])==3
    other.close()


def test_routine_rejects_unknown_shell_and_destructive_steps(brain,registry):
    for step in ('powershell arbitrary command','open powershell','delete file','shutdown computer'):
        assert brain.routines.save('unsafe',['open chrome',step]) is None
    assert brain.memory.counts()['routines']==0
    registry.run.assert_not_called()


def test_routine_changed_step_preflights_before_any_execution(brain,registry):
    brain.routines.save('mode',['open chrome','open spotify'])
    brain.commands.pop('open spotify')
    result=brain.routines.execute('mode',lambda command:brain._execute(command,1,{},exact=True,confirmed=True))
    assert not result['success']
    registry.run.assert_not_called()


def test_routine_stops_on_failure_and_tracks_failure(brain,registry):
    brain.routines.save('mode',['open chrome','open spotify'])
    registry.run.return_value='Could not launch app'
    registry.run.side_effect=None
    result=brain.routines.execute('mode',lambda command:brain._execute(command,1,{},exact=True,confirmed=True))
    assert not result['success']
    assert registry.run.call_count==1
    assert brain.routines.get('mode')['failure_count']==1


def test_routine_update_and_disable(brain,registry):
    first=brain.routines.save('mode',['open chrome'])
    assert brain.routines.save('mode',['open spotify'])==first
    assert len(brain.routines.get('mode')['steps'])==1
    brain.memory.db.execute('UPDATE routines SET enabled=0 WHERE id=?',(first,))
    assert 'disabled' in brain.process('mode')['text']
    registry.run.assert_not_called()


def test_pending_routine_update_requires_new_confirmation(brain,registry):
    brain.routines.save('mode',['open chrome'])
    brain.confirmer=Mock(return_value='pending')
    brain.process('mode')
    brain.routines.save('mode',['open spotify'])
    assert 'changed after' in approve_pending(brain)
    registry.run.assert_not_called()


def test_malformed_or_sensitive_routine_steps_never_display_or_execute(brain,registry):
    brain.routines.save('mode',['open chrome'])
    for payload in ('not-json','{"command_phrase":"password=synthetic-value"}'):
        with brain.memory.db:
            brain.memory.db.execute('UPDATE routine_steps SET action_args=?',(payload,))
        result=brain.process('mode')
        assert result['intent']=='clarification'
        assert 'synthetic-value' not in result['text']
        registry.run.assert_not_called()


def test_project_routine_survives_restart(brain,registry,tmp_path):
    project=tmp_path/'project'
    project.mkdir()
    brain.process('When I say Alpha project, use '+str(project))
    assert brain.routines.save('project mode',['open chrome','open my Alpha project'])
    store=AlphaMemory(brain.memory.db.execute('PRAGMA database_list').fetchone()[2])
    try:
        second=OfflineBrain(store,Mock(),settings={})
        configure_app_commands(second,registry,['chrome'])
        second.confirmer=Mock(return_value='pending')
        second.process('project mode')
        approve_pending(second)
        assert second.routines.get('project mode')['success_count']==1
    finally:
        store.close()


def test_repeat_safe_action_and_confirmed_editor(brain,registry):
    brain.process('open chrome')
    brain.process('again')
    assert registry.run.call_count==2
    brain.confirmer=Mock(return_value='pending')
    brain.process('open vscode')
    approve_pending(brain)
    registry.run.reset_mock()
    assert brain.process('do that again')['confirmation_required']
    registry.run.assert_not_called()
    approve_pending(brain)
    assert registry.run.call_count==1


def test_repeat_routine_requires_fresh_gate(brain,registry):
    brain.routines.save('mode',['open chrome','open spotify'])
    brain.confirmer=Mock(return_value='pending')
    brain.process('mode')
    approve_pending(brain)
    registry.run.reset_mock()
    assert brain.process('repeat that')['confirmation_required']
    registry.run.assert_not_called()
    approve_pending(brain)
    assert registry.run.call_count==2


def test_unsafe_repeat_is_not_executed(brain,registry):
    brain.confirmer=Mock(return_value='pending')
    brain.process('open spotify')
    brain.process('close spotify')
    approve_pending(brain)
    registry.run.reset_mock()
    assert brain.process('again')['intent']=='clarification'
    registry.run.assert_not_called()


def test_negative_action_feedback_blocks_repeat(brain,registry):
    brain.process('open chrome')
    brain.process('open chrome')
    brain.process('that was wrong')
    registry.run.reset_mock()
    assert brain.process('again')['intent']=='clarification'
    registry.run.assert_not_called()


def test_pattern_detection_never_saves_or_executes(brain,registry):
    for _ in range(3):
        brain.process('open chrome')
        brain.process('open spotify')
    suggestion=brain.process('suggest a routine')
    assert 'frequently' in suggestion['text']
    assert brain.memory.counts()['routines']==0
    assert registry.run.call_count==6
    brain.provider.generate.assert_not_called()


def test_pattern_detector_bounds_and_gaps():
    detector=PatternDetector(max_history=6)
    for _ in range(3):
        detector.record('a')
        detector.record('b')
    assert detector.suggestions()[0]['steps']==['a','b']
    detector.record('c')
    assert len(detector.history)==6
    detector=PatternDetector(max_gap=1)
    for _ in range(3):
        detector.history.extend([('a',0),('b',100)])
    assert detector.suggestions()==[]


def test_ranking_prefers_verified_over_high_candidate(memory):
    memory.store_solution('missing python module PyQt6','An unverified answer.',confidence=1)
    good=memory.store_solution('module missing PyQt6','A verified reusable answer.',source='user',confidence=.85)
    result=memory.search('PyQt6 module missing',eligible_only=True)
    assert result['id']==good
    assert result['confidence']>=.80
    assert 'score=' in result['reason']


def test_consolidation_preserves_verified_and_conflicts(memory):
    good=memory.store_solution('local database storage','Keep this user-confirmed answer.',source='user',confidence=.9)
    memory.store_solution('local database storage','A conflicting proposed answer.')
    first=memory.store_solution('missing module PyQt6','Install the package into your active environment.')
    second=memory.store_solution('module missing PyQt6','Install the package into your active environment.')
    maintenance=MemoryConsolidator(memory)
    assert maintenance.analyze()['conflicts']
    assert maintenance.deduplicate_candidates()==1
    assert memory.get_knowledge(good)['verification_status']=='verified'
    assert all(memory.get_knowledge(item) for item in (first,second))


def test_consolidation_detects_stale_and_promotes_only_verified(memory):
    stale=memory.store_solution('old candidate','An old unverified answer.')
    good=memory.store_solution('successful answer','A confirmed reusable answer.',source='user',confidence=.9)
    with memory.db:
        memory.db.execute("UPDATE learned_knowledge SET updated_at='2000-01-01 00:00:00' WHERE id=?",(stale,))
        memory.db.execute('UPDATE learned_knowledge SET success_count=3 WHERE id IN (?,?)',(stale,good))
    maintenance=MemoryConsolidator(memory)
    assert stale in maintenance.analyze()['stale_candidates']
    assert maintenance.trust_successful()==1
    assert memory.get_knowledge(good)['verification_status']=='trusted'
    assert memory.get_knowledge(stale)['verification_status']=='candidate'


@pytest.mark.parametrize('payload',[
    'password=synthetic-value','my password is synthetic-value','api_key=synthetic-value',
    'token: synthetic-value','Cookie: session=synthetic-value',
    'authentication_cookie=synthetic-value','credentials=synthetic-value',
    'access_token=synthetic-value','refresh_token=synthetic-value','auth_cookie=synthetic-value',
    '-----BEGIN '+'PRIVATE KEY----- synthetic',
])
def test_secret_all_boundaries_and_inspection(brain,payload):
    assert brain.memory.store_solution('question',payload) is None
    assert not brain.memory.add_alias('name',payload)
    assert brain.routines.save(payload,['open chrome']) is None
    assert brain.process(payload)['intent']=='sensitive_input'
    # Simulate legacy data: inspection re-filters it rather than revealing it.
    item=brain.memory.store_solution('old answer','An otherwise harmless answer.')
    brain.memory.db.execute('UPDATE learned_knowledge SET response=? WHERE id=?',(payload,item))
    result=brain.process('Alpha, show your recent memories.')
    assert payload not in result['text']
    brain.provider.generate.assert_not_called()


def test_inspection_is_local_and_preserves_feedback_target(brain):
    brain.process('missing PyQt6')
    brain.process('remember this')
    for command in ('Alpha, what did you learn today?','Alpha, what do you remember about PyQt6?',
                    'what aliases do you know?','what routines do you know?',
                    'Alpha, why did you choose that solution?'):
        assert brain.process(command)['source']=='local'
    assert brain.process('that worked')['intent']=='feedback'
    assert brain.memory.inspect()[0]['verification_status']=='verified'
    assert brain.provider.generate.call_count==1


def test_sensitive_teacher_response_not_stored_or_displayed(brain):
    brain.provider.generate.return_value={'ok':True,'text':'password=synthetic-value'}
    assert brain.process('question')['intent']=='sensitive_input'
    assert brain.memory.counts()['knowledge']==0
    assert brain.context.recent() is not None
    assert 'synthetic-value' not in brain.context.summary()


def test_migration_is_additive_idempotent_and_retains_legacy_data(tmp_path):
    path=tmp_path/'legacy.db'
    db=sqlite3.connect(path)
    db.executescript('''CREATE TABLE learned_knowledge (
        id INTEGER PRIMARY KEY,topic TEXT NOT NULL,trigger_text TEXT NOT NULL,
        normalized_trigger TEXT NOT NULL UNIQUE,response TEXT NOT NULL,solution TEXT NOT NULL,
        source TEXT NOT NULL,confidence REAL NOT NULL,success_count INTEGER NOT NULL DEFAULT 0,
        failure_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,last_used_at TEXT);
        INSERT INTO learned_knowledge(topic,trigger_text,normalized_trigger,response,solution,source,confidence,success_count)
        VALUES ('test','topic','topic','Existing answer preserved.','Existing answer preserved.','ollama',0.9,2);''')
    db.close()
    for _ in range(3):
        store=AlphaMemory(path)
        assert store.get_knowledge(1)['response']=='Existing answer preserved.'
        assert store.get_knowledge(1)['verification_status']=='verified'
        assert store.db.execute('PRAGMA user_version').fetchone()[0]==3
        assert store.counts()['knowledge']==1
        store.close()


def test_workspace_permission_context_not_model_parameters():
    with patch('actions.local_workspace._perform',return_value='Done') as perform, patch('actions.local_workspace.confirm.request',return_value='pending') as gate:
        result=_call_handler(local_workspace,{'operation':'run_project','target':'local','alpha_confirmed':True},{})
        assert result=='pending'
        perform.assert_not_called()
        assert _call_handler(local_workspace,{'operation':'run_project','target':'local'}, {'alpha_confirmed':True})=='Done'
        perform.assert_called_once()


@pytest.mark.skipif(sys.platform!='win32',reason='Windows action adapter')
def test_workspace_sends_no_shell_or_force_kill(tmp_path):
    from actions.local_workspace import _perform
    (tmp_path/'main.py').write_text('print("test")')
    with patch('actions.local_workspace.subprocess.Popen') as start:
        assert _perform('run_project',str(tmp_path)).startswith('Started project')
        assert start.call_args.args[0][-1]==str(tmp_path/'main.py')
        assert start.call_args.kwargs.get('shell',False) is False
    with patch('actions.local_workspace.os.startfile',create=True) as open_path:
        assert _perform('open_project',str(tmp_path)).startswith('Opened project')
        open_path.assert_called_once_with(str(tmp_path.resolve()))


@pytest.mark.skipif(sys.platform!='win32',reason='Windows window-message adapter')
def test_close_targets_known_process_windows_without_terminating():
    from actions.local_workspace import _close_windows_app
    user32=Mock()
    user32.IsWindowVisible.return_value=True
    user32.PostMessageW.return_value=True
    def owner(hwnd,pid):
        pid._obj.value=42 if hwnd==100 else 99
    user32.GetWindowThreadProcessId.side_effect=owner
    user32.EnumWindows.side_effect=lambda callback,param: (callback(100,param),callback(200,param))
    process=SimpleNamespace(info={'name':'chrome.exe','pid':42})
    with patch('ctypes.WinDLL',return_value=user32), patch('psutil.process_iter',return_value=[process]):
        assert _close_windows_app('chrome').startswith('Sent close request')
        user32.PostMessageW.assert_called_once_with(100,0x0010,0,0)
    assert _close_windows_app('unknown & command')=='This application cannot be targeted safely.'


def test_ollama_model_unavailable_reports_options_without_fallback():
    from core.providers.ollama_provider import OllamaProvider
    session=Mock()
    session.get.return_value.status_code=200
    session.get.return_value.json.return_value={'models':[{'name':'installed:latest'}]}
    result=OllamaProvider(model='missing',session=session).generate('question')
    assert not result['ok']
    assert 'installed:latest' in result['text']
    session.post.assert_not_called()
