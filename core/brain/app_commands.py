"""Conservative adapter to existing actions; no free-form shell/app arguments."""
from core.offline_brain import Command
from core.brain.intent_engine import normalize


def configure_app_commands(brain, registry, known_apps, player=None, confirmer=None):
    brain.confirmer = confirmer
    brain.register_command('alpha brain status', lambda ctx: 'ALPHA offline core and local memory are ready.', 'brain_status')
    # Existing app aliases supply the targets; terminal interpreters require confirmation.
    interpreters = {'terminal', 'cmd', 'powershell', 'git', 'code', 'vscode', 'visual studio code'}
    if registry.has('open_app'):
        for app in known_apps:
            def handler(ctx, app=app):
                text = registry.run('open_app', {'app_name':app}, {'player':player})
                return {'text':text, 'success':isinstance(text,str) and text.startswith('Opened ')}
            for verb in ('open', 'launch', 'start'):
                brain.register_command(f'{verb} {app}', handler, 'open_app', dangerous=app in interpreters,
                    repeatable=True, routine_allowed=app not in {'terminal','cmd','powershell','git'},
                    action_name='open_app',action_args={'app_name':app},entity={'kind':'app','target':app})
        # Canonicalize spoken spelling without adding a second application inventory.
        if 'vscode' in known_apps:
            for verb in ('open','launch','start'):
                brain.commands[f'{verb} vs code'] = brain.commands[f'{verb} vscode']
    if registry.has('local_workspace'):
        for app in known_apps:
            def close(ctx, app=app):
                text = registry.run('local_workspace',{'operation':'close_app','target':app},
                                    {'player':player,'alpha_confirmed':ctx.get('_alpha_confirmed',False)})
                return {'text':text,'success':isinstance(text,str) and text.startswith('Sent close request')}
            brain.register_command('close '+app,close,'close_app',dangerous=True,repeatable=False,
                action_name='local_workspace',action_args={'operation':'close_app','target':app},entity={'kind':'app','target':app})
        def project_command(verb, path):
            from pathlib import Path
            if verb not in ('open','launch','start','run') or path.startswith(('\\\\','//')) or not Path(path).is_absolute() or not Path(path).is_dir():
                return None
            operation = 'run_project' if verb=='run' else 'open_project'
            phrase = ('run ' if verb=='run' else 'open ')+path
            def run(ctx):
                text = registry.run('local_workspace',{'operation':operation,'target':path},
                                    {'player':player,'alpha_confirmed':ctx.get('_alpha_confirmed',False)})
                return {'text':text,'success':isinstance(text,str) and text.startswith(('Opened project','Started project'))}
            command = Command(phrase,run,operation,dangerous=verb=='run',repeatable=verb!='run',
                routine_allowed=verb!='run',action_name='local_workspace',
                action_args={'operation':operation,'target':path},entity={'kind':'project','target':path})
            brain.commands[normalize(phrase)] = command
            return command
        brain.adaptive.project_factory = project_command
    # Other actions/plugins remain in the current dispatcher until explicit intent
    # parsers and safety metadata are supplied. Never infer parameters from descriptions.
