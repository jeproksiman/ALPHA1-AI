"""Manual, non-destructive local model smoke test; never prints prompts/vectors."""
import sys
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory
from statistics import median

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config_manager import get_offline_brain_settings
from core.providers.ollama_provider import OllamaProvider
from core.offline_brain import OfflineBrain
from memory.alpha_memory import AlphaMemory


def local_latencies(provider):
    with TemporaryDirectory() as directory:
        memory=AlphaMemory(Path(directory)/'bench.db')
        try:
            brain=OfflineBrain(memory,provider,settings={})
            brain.register_command('local status',lambda ctx:'Offline core ready.')
            memory.add_alias('status alias','local status')
            memory.store_solution('What is SQLite','SQLite is a local relational database.',source='user',confidence=.9)
            for label,text in [('exact','local status'),('alias','status alias'),('verified memory','What is SQLite')]:
                samples=[]
                for _ in range(10):
                    start=perf_counter();brain.process(text);samples.append((perf_counter()-start)*1000)
                print(f'{label} median: {median(samples):.2f}ms')
            for label in ('first semantic lookup','cached semantic lookup'):
                start=perf_counter()
                brain.semantic.search('Where can I keep records locally?')
                print(f'{label}: {(perf_counter()-start)*1000:.2f}ms')
        finally:
            memory.close()


def main():
    cfg = get_offline_brain_settings()
    provider = OllamaProvider(cfg['ollama_url'],cfg['ollama_model'],cfg['ollama_enabled'],timeout=90,
        fast_model=cfg['ollama_fast_model'],embedding_model=cfg['embedding_model'])
    status = provider.model_status()
    print('Ollama reachable:',status['available'])
    for role,info in status['roles'].items():
        print(role+': '+info['model']+' '+('ready' if info['ready'] else 'unavailable'))
    if not all(info['ready'] for info in status['roles'].values()):
        print('Installed local alternatives: '+(', '.join(status['installed']) or 'none'))
        return 1
    start = perf_counter()
    generated = provider.chat([{'role':'user','content':'Reply with one word: ready.'}],max_tokens=32,think=False)
    print(f"Primary generation: {'passed' if generated['ok'] else 'failed'} ({perf_counter()-start:.2f}s)")
    start = perf_counter()
    embedded = provider.embed('A local SQLite memory stores verified answers.')
    print(f"Embedding: {'passed' if embedded['ok'] else 'failed'} ({perf_counter()-start:.2f}s)")
    if generated['ok'] and embedded['ok']:
        local_latencies(provider)
    return 0 if generated['ok'] and embedded['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
