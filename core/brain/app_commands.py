"""Conservative adapter to existing actions; no free-form shell/app arguments."""
from core.offline_brain import Command


def configure_app_commands(brain, registry, known_apps, player=None, confirmer=None):
    brain.confirmer = confirmer
    brain.register_command('alpha brain status', lambda ctx: 'ALPHA offline core and local memory are ready.', 'brain_status')
    # Existing app aliases supply the targets; terminal interpreters require confirmation.
    interpreters = {'terminal', 'cmd', 'powershell', 'git', 'code', 'vscode', 'visual studio code'}
    if registry.has('open_app'):
        for app in known_apps:
            handler = lambda ctx, app=app: registry.run('open_app', {'app_name': app}, {'player': player})
            for verb in ('open', 'launch', 'start'):
                brain.register_command(f'{verb} {app}', handler, 'open_app', dangerous=app in interpreters)
    # Other actions/plugins remain in the current dispatcher until explicit intent
    # parsers and safety metadata are supplied. Never infer parameters from descriptions.
