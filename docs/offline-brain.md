# ALPHA v2 offline brain — Phase 1

The existing application has Gemini/Deepgram live voice, action and plugin
registries, an Ollama/OpenAI-compatible helper client (`core/llm_client.py`),
JSON personal memory, and a shared confirmation gate. These remain available.
The new SQLite store holds learned solutions rather than duplicating personal
profile memory. No existing JSON memory is migrated or deleted.

## Integration and routing

The text input box uses `OfflineBrain.process(text, context=None)`. Upload,
clipboard, quiz, remote dashboard, and live voice paths retain their existing
handling. Prefix a typed request with `live:` to use existing live tool/plugin
handling, or disable the brain to restore all previous typed behavior. A live
session is needed for that escape path. Offline text responses appear in the
activity log; offline speech recognition and synthesis are deferred.

Initialization occurs before waiting for a cloud API key. The existing setup
dialog and voice connection still require their normal configuration; this
phase does not redesign onboarding or remove cloud audio dependencies.

Routing order: exact registered command, explicit skill hook, command alias,
exact learned memory, context hook, fuzzy memory/command match, Ollama fallback.
Local confidence must reach 0.80 by default. Exact registered app commands reuse
the existing action registry and its known app aliases. Arbitrary app arguments
are not accepted. Launch/start/open phrases share those targets. Shell/editor
launches, aliases to actions, and fuzzy action matches require the existing HUD
confirmation gate; an unavailable gate refuses execution. Other actions/plugins
can be integrated later through explicit parsers; descriptions are not parsers.
Context and skill hooks are extension points, not a full natural-language router.

The result contains `text`, `source` (`local`, `memory`, `ollama`), `confidence`,
`intent`, and `learned`. Memory results also expose `knowledge_id` for feedback.
Learned text is only returned to the user: it is never parsed into tool calls.

## Ollama

The fallback uses the documented local `/api/tags` and non-streaming `/api/chat`
endpoints. It accepts only loopback HTTP URLs, refuses redirects and remote URLs,
ignores proxy environment settings, and uses bounded connection/read timeouts.
No model, service, or download is automatically started. If no model is configured,
the first alphabetically sorted installed model is used. Disabled/unreachable
services, absent models, invalid replies, and timeouts produce a local message.
The new adapter avoids the older helper client's service restart and tool-call
behavior; it reuses its existing configuration values where appropriate.

API references: https://docs.ollama.com/api/chat and https://docs.ollama.com/api/tags

## Configuration

`memory/config_manager.py` reads environment overrides, then the corresponding
lowercase keys from the existing ignored `config/api_keys.json`. It never writes
or prints credentials. `.env.example` documents the options; `.env` is not loaded
automatically. Set environment variables or the existing JSON settings explicitly.

| Environment variable | Default |
| --- | --- |
| ALPHA_OFFLINE_BRAIN_ENABLED | true |
| ALPHA_OLLAMA_ENABLED | true |
| ALPHA_OLLAMA_URL | http://127.0.0.1:11434 |
| ALPHA_OLLAMA_MODEL | empty: detect installed models |
| ALPHA_LOCAL_CONFIDENCE_THRESHOLD | 0.80 (supported range 0.80–1.00) |
| ALPHA_AUTO_LEARN_OLLAMA | false |

Legacy `llm_url`/`llm_model` are reused when `llm_provider` is Ollama. An
OpenAI-compatible backend's settings do not redirect this fallback.

## Memory and learning

Default path: `memory/alpha_memory.db`, ignored along with SQLite sidecars and
existing runtime personal JSON memory. SQLite uses parameterized SQL, a lock for
worker access, a unique normalized trigger, and indexed exact lookups. Fuzzy
search scans at most 1,000 confidence-ranked records; scaling that is Phase 2.

| Table | Fields |
| --- | --- |
| learned_knowledge | id, topic, trigger_text, normalized_trigger, response, solution, source, confidence, success_count, failure_count, created_at, updated_at, last_used_at |
| aliases | id, phrase, target, confidence, created_at, updated_at |
| corrections | id, original_input, wrong_result, corrected_result, created_at |
| interaction_history | id, user_input, response, source, intent, confidence, success, created_at |

`Learner.learn_from_solution(...)` extracts a bounded reusable answer. Ollama
candidates start at 0.60 and are capped at 0.70 on insertion. Repeated triggers
return the existing ID without overwriting feedback or promoting confidence.
Explicit `record_success(id)` adds 0.10; `record_failure(id)` subtracts 0.25,
clamped to [0, 1]. Natural-language feedback parsing is deferred.

Auto-learning is off. Enabling it stores only filtered candidate answers; it
does not verify them. Interaction history is opt-in through an explicit method,
not raw logging on every request. Corrections are recorded explicitly and do not
silently overwrite a learned answer.

Secret/credential patterns and dangerous command patterns are checked before
every persistence entry point. Oversized output and terminal-sized dumps are
rejected. This is a basic filter, not a guarantee that every sensitive phrase is
recognized. Keep auto-learning disabled for sensitive workflows. Sensitive input
matching the filter is neither persisted nor sent to Ollama. Existing live/cloud
logging and legacy personal-memory behavior are unchanged by this phase.

## Validation and Phase 2

Run `python -m unittest discover -s tests -v` or `python -m pytest`, then
`python -m compileall -q .` and `git diff --check`. Tests use temporary databases
and mocked HTTP and require no Ollama service or GUI dependencies.

Phase 2: explicit safety metadata and parameter parsers for more registry actions,
user-facing feedback and candidate review, contextual routines, offline voice,
bounded retention and indexed token search. Confirm memory quality before
expanding automation. No vectors, embeddings, model dumps, self-modifying code,
automatic model downloads, or unrestricted autonomy are included here.
