"""Narrow Windows workspace actions; all changing actions use the shared gate."""
import os
import re
import sys
import subprocess
from pathlib import Path
from core import confirm


def _project_path(target):
    path = Path(target)
    if not path.is_absolute() or str(target).startswith(('\\\\','//')) or not path.is_dir():
        raise ValueError('Use an existing absolute local directory')
    return path.resolve()


def _close_windows_app(target):
    """Send WM_CLOSE to windows owned by an exact known executable; never force-kill."""
    from actions.open_app import _APP_ALIASES
    from ctypes import wintypes
    import ctypes
    import psutil
    app = _APP_ALIASES.get(target,{}).get('Windows','')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',app):
        return 'This application cannot be targeted safely.'
    wanted = app if app.lower().endswith('.exe') else app+'.exe'
    pids = set()
    for process in psutil.process_iter(['pid','name']):
        if (process.info['name'] or '').casefold()==wanted.casefold():
            pids.add(process.info['pid'])
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type,wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
    count = 0
    @callback_type
    def callback(hwnd, param):
        nonlocal count
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            if user32.PostMessageW(hwnd,0x0010,0,0):
                count += 1
        return True
    user32.EnumWindows(callback,0)
    return 'Sent close request to '+target+'.' if count else 'No matching application window was found.'


def _perform(operation, target):
    if sys.platform != 'win32':
        return 'Workspace actions currently require Windows.'
    try:
        if operation=='close_app':
            return _close_windows_app(target)
        path = _project_path(target)
        if operation=='open_project':
            os.startfile(str(path))
            return 'Opened project directory.'
        if operation=='run_project':
            entry = path/'main.py'
            if not entry.is_file() or entry.is_symlink():
                return 'Project has no regular main.py entry point.'
            if getattr(sys,'frozen',False):
                return 'Configure a Python interpreter before running projects from a packaged app.'
            subprocess.Popen([sys.executable,str(entry)],cwd=str(path),
                             creationflags=subprocess.CREATE_NO_WINDOW)
            return 'Started project main.py. Completion has not been verified.'
        return 'Unsupported workspace operation.'
    except Exception:
        return 'Workspace action could not complete.'


def local_workspace(parameters, player=None, alpha_confirmed=False):
    operation = parameters.get('operation')
    target = parameters.get('target','')
    if operation not in ('open_project','run_project','close_app'):
        return 'Unsupported workspace operation.'
    if operation in ('run_project','close_app') and not alpha_confirmed:
        return confirm.request('workspace_'+operation,operation+' '+target,
            'This may close unsaved work or execute project code. Confirm the explicit target.',
            lambda:_perform(operation,target))
    return _perform(operation,target)


TOOL = {
    'name':'local_workspace',
    'description':'Open an explicit local project directory, or request confirmation to run its main.py or close a known application window.',
    'parameters':{'type':'OBJECT','properties':{
        'operation':{'type':'STRING','enum':['open_project','run_project','close_app']},
        'target':{'type':'STRING'}},'required':['operation','target']},
    'handler':local_workspace,
}
