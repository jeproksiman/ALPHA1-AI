"""Read-only local structured planning smoke; never executes a real action."""
import contextlib
import io
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.action_loader import discover_actions
from core.brain.action_planner import ActionPlanner
from core.providers.ollama_provider import OllamaProvider
from memory.config_manager import get_offline_brain_settings


def main():
    cfg = get_offline_brain_settings()
    provider = OllamaProvider(cfg['ollama_url'], cfg['ollama_model'], cfg['ollama_enabled'],
        fast_model=cfg['ollama_fast_model'], embedding_model=cfg['embedding_model'], timeout=90)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        registry = discover_actions(Path(__file__).resolve().parents[1] / 'actions', logger=lambda message: None)
    executed = []
    def synthetic_execution(name, args, confirmed=False):
        executed.append(name)
        return 'Synthetic retrieved result: the example announcement occurred today. This is a test fixture.'
    planner = ActionPlanner(provider, registry, executor=synthetic_execution)
    started = perf_counter()
    result = planner.respond('Search the web for an example announcement today.', {})
    passed = executed == ['web_search'] and result and result['intent'] == 'action_result'
    print('Local structured selection + primary result summary:', 'passed' if passed else 'failed')
    print('Registered actions:', len(registry.names()))
    print('Elapsed seconds:', round(perf_counter() - started, 2))
    provider.session.close()
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
