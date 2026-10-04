# ALPHA v2 offline brain — Phases 1 and 2

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
launches, aliases to actions, contextual actions, routines, and fuzzy action matches require the existing HUD
confirmation gate; an unavailable gate refuses execution. Other actions/plugins
can be integrated later through explicit parsers; descriptions are not parsers.
Skill hooks remain extension points. Phase 2 adds bounded conversational context
and deterministic feedback, alias, routine, and inspection grammars ahead of this
ordinary request route. This is not a general natural-language action parser.

The result contains `text`, `source` (`local`, `memory`, `ollama`), `confidence`,
`intent`, and `learned`. Memory results also expose `knowledge_id` and `reason`
for feedback and explanation. Action results add structured target/action metadata.
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
| ALPHA_OLLAMA_MODEL | gemma4:e4b |
| ALPHA_OLLAMA_FAST_MODEL | qwen3.5:0.8b |
| ALPHA_SEMANTIC_MEMORY | true |
| ALPHA_EMBEDDING_PROVIDER | ollama |
| ALPHA_EMBEDDING_MODEL | nomic-embed-text:latest |
| ALPHA_LOCAL_CONFIDENCE_THRESHOLD | 0.80 (supported range 0.80–1.00) |
| ALPHA_AUTO_LEARN_OLLAMA | false |

The offline brain now uses its explicit `ALPHA_*` settings and the role defaults
above. Legacy `llm_url`/`llm_model` settings still belong to the existing helper
client and do not override these roles. An OpenAI-compatible backend's settings
do not redirect this fallback. The example environment file documents values;
defaults already activate the requested local models without loading `.env`.

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
Phase 2 feedback uses the rules described below; Phase 1's standalone confidence
helper remains available for compatibility, but database feedback uses adaptive
rules and records the actual change.

Auto-learning is off. Enabling it stores only filtered candidate answers; it
does not verify them. Interaction history is opt-in through an explicit method,
not raw logging on every request. Corrections are recorded explicitly and do not
silently overwrite a learned answer. Explicit replacement corrections preserve
the previous answer in correction history.

Secret/credential patterns and dangerous command patterns are checked before
every persistence entry point. Oversized output and terminal-sized dumps are
rejected. This is a basic filter, not a guarantee that every sensitive phrase is
recognized. Keep auto-learning disabled for sensitive workflows. Sensitive input
matching the filter is neither persisted nor sent to Ollama. Existing live/cloud
logging and legacy personal-memory behavior are unchanged by this phase.

## Phase 2: verified feedback

The new `FeedbackEngine` recognizes exact positive phrases (`that worked`,
`it worked`, `perfect`, `that fixed it`), negative phrases (`that didn't work`,
`wrong`, `that was wrong`, `it failed`, `don't use that solution`), and explicit
learning (`remember this`, `remember how we fixed this`, `learn this solution`).
An optional `Alpha,` prefix is supported. No model is needed for this grammar.

Feedback attaches to the latest eligible session interaction. Inspection and
alias/routine teaching do not replace that target. Failed fallback calls and
pending confirmations form barriers: feedback cannot accidentally verify an older
answer. Context expires after 15 minutes. Each interaction accepts a feedback
kind once, with unique confidence-event receipts preventing duplicate promotion.

An Ollama response stays in bounded working context by default. Explicit feedback
stores it only after extraction/filtering. `remember this` creates a candidate,
not verified knowledge. Negative feedback retains a filtered failed answer for
diagnostics. Auto-learning still defaults to **false**.

| Event | Confidence/state effect |
| --- | --- |
| New teacher candidate | 0.60, candidate |
| Explicit success | +0.15; first verification has a 0.85 floor |
| Confirmed successful reuse via API | +0.05 |
| Explicit failure | -0.20; demote to candidate |
| Automatic failure via API | -0.10; demote to candidate |
| Three explicit successes | trusted |
| Two failures or explicit “don't use” | rejected |
| Explicit replacement correction | verified at 0.85; preserve old answer |

All confidence is clamped to [0, 1]. A read/recommendation increments `use_count`
but is **not** proof of success and does not automatically raise confidence.
`confidence_events` stores actual deltas and reasons, including the verification
floor. Rejected knowledge is never silently deleted or revived by positive
feedback; an explicit corrected replacement is required.

Use `the correct solution is ...` or `instead use ...` to replace the recent
answer. Conflicting teacher answers are saved as diagnostic proposals and cannot
verify a different previously saved response. Aliases correct application targets
instead of turning text solutions into executable actions.

## Phase 2: aliases, context, and repeat

Examples:

```
When I say editor, I mean VS Code.
When I say music, open Spotify.
No, by browser I mean Chrome.
When I say Alpha project, use C:\Users\TUF\Desktop\alphav55
open editor
open my Alpha project
run it
```

Aliases persist at 0.98 confidence and explicit reteaching updates their target.
Targets must be registered applications/actions or existing absolute local
directories. Alias cycles, shell fragments, unknown targets, network paths, and
sensitive values are refused. `VS Code` resolves to the existing `vscode` target.
An alias never creates arbitrary code.

`ContextManager` keeps up to 20 requests/responses and bounded entity metadata in
RAM. Successful registered actions identify the recent app/project. `close it`,
`open it`, and `run it` use that recent target and **always require confirmation**.
An intervening unrelated answer, pending/failed action, expired context, or absent
target prompts clarification. Bare pronouns ask for the intended action.

`again`, `do that again`, `repeat that`, and `same thing` find the last successful
registered action/routine within the context window. Safe repeatable launches can
repeat locally; editor/interpreter launches retain their confirmation gate.
Closing apps and running project code are not repeatable. Routines require a
fresh confirmation on every invocation, including repeats.

The `local_workspace` action integrates into the existing registry. It opens a
directory without a shell, starts only that project's `main.py` through a Python
argument list after confirmation, and closes a known Windows app by sending
`WM_CLOSE` to matching process-owned windows after confirmation. It never force
kills a process. A close request or process launch is reported as a request/start,
not a verified completed outcome. Packaged apps refuse project execution until
an interpreter integration exists. Other platforms return a helpful message.

The action registry accepts an internal confirmation context marker. It is not
part of the model-visible tool schema, and a similarly named tool argument cannot
authorize execution. Deferred callbacks update context only after execution.

## Phase 2: routines and patterns

```
When I say coding mode: open VS Code, open Chrome, open my Alpha project.
coding mode
suggest a routine
```

Routines contain 1–10 ordered registered action steps. Creation is an explicit
user instruction; execution goes through the shared HUD gate. Known app launches
and project directory opening are permitted. Shell launches, project execution,
deletion, shutdown, registry/security/credential changes, and unknown steps are
excluded. All steps are revalidated before the first executes. Saved targets are
fixed at creation; reteach a routine to adopt a changed alias. Registry changes
that alter a saved target require review. The confirmation displays the ordered
targets; changing a routine while confirmation is pending invalidates that request.
A failed step stops the routine;
success/failure counts persist. Repeated teaching updates the routine in place.

`PatternDetector` retains at most 60 successful action events in RAM, finds
non-overlapping two/three-step sequences seen at least three times, and excludes
widely separated steps. `suggest a routine` displays local suggestions. Detection
does **not** save or execute a routine; the user must explicitly teach its name
and steps. Pattern history does not survive restarting the app.

## Phase 2: ranking, inspection, and maintenance

Only verified/trusted knowledge is eligible for local answer routing. Rejected
items score zero; candidates cannot exceed 0.70. Exact answers retain stored
confidence. Fuzzy ranking combines token/character matching, verification status,
successes, bounded usage, and recent updates. Each result describes those factors
in `reason`. A small deterministic missing-module vocabulary supports
`ModuleNotFoundError: PyQt6` / `PyQt6 module missing`; mismatching package versions
and opposite action words do not match. No embedding/vector service is used.

Local inspection phrases include:

- `Alpha, what did you learn today?` (Manila date)
- `Alpha, what do you remember about PyQt6?`
- `Alpha, show your recent memories.`
- `Alpha, what aliases have I taught you?`
- `Alpha, what routines do you know?`
- `Alpha, why did you choose that solution?`

Inspection is bounded and re-filters stored rows, including legacy data, before
display. Sensitive teacher responses are discarded from output/context/storage.
The basic secret filter also covers authentication cookies and common access/
refresh-token field names. It is still a pattern filter, not a universal detector.

`MemoryConsolidator(memory).analyze()` reports near duplicates, conflicts, stale
candidates (30 days), and frequently successful items without invoking Ollama.
`deduplicate_candidates()` marks identical near-duplicate candidates rejected,
retaining their records. Exact duplicates are prevented at insertion.
`trust_successful()` promotes only already verified successful knowledge.
Maintenance is explicit through this API; there is no background destructive job.
Inspection/consolidation operate on bounded windows (up to 1,000 inspection rows,
50 consolidation entries). Larger-store maintenance is a later extension.

## Phase 2: schema and migration

Schema version 2 adds `verification_status`, `use_count`, `last_feedback`, and
`last_feedback_at` to learned knowledge. New tables are `confidence_events`,
`routines`, `routine_steps`, and `knowledge_conflicts`. Routine steps store JSON
arguments and registered action identities, never Python source.

Migration checks columns and uses additive `ALTER TABLE` / `CREATE IF NOT EXISTS`.
It preserves IDs, responses, feedback counts, existing tables, and database files.
Legacy successful knowledge becomes verified/trusted only where prior failures
do not contradict it. Reopening the database does not reset verification state.
All runtime SQLite databases and sidecars remain ignored by Git.

Ollama's prompt describes its teacher role and ALPHA's existing local actions.
When an explicitly configured model is absent, available installed names are
reported; ALPHA does not silently select another model or download one. With no
configured model (an explicit empty setting), deterministic installed-model
selection is preserved, limited to local non-embedding models. The application
default is now explicitly `gemma4:e4b`.

## Local model roles and optional semantic memory

The primary fallback is **gemma4:e4b**. **qwen3.5:0.8b** is available only through
the explicit `provider.lightweight(text, task=...)` API for short classification,
summarization, tags, and choosing among a bounded option list. It is not called
automatically for known commands, feedback, learning, or ordinary reasoning.
Fast-model failure returns a helpful failure without silently switching roles.

**nomic-embed-text:latest** generates local embeddings through `/api/embed`.
Generation and embedding roles have separate configured models. Startup fetches
the local model list once and reports each role without loading models or building
an index. Missing roles report installed local alternatives; no downloads occur.

The provider accepts only loopback HTTP, disables environment proxies, refuses
redirects and embedded URL credentials, and rejects cloud model names and models
advertised with remote-host/model metadata. These rules also apply to fast and
embedding requests. No external inference opt-out is introduced in this phase.
The existing live voice/cloud mode is separate and remains available through its
previous paths; this guarantee covers the offline-brain Ollama traffic.

Request priority after deterministic feedback/inspection/teaching controls:
exact command → registered action → alias → routine → verified exact memory →
recent context → deterministic fuzzy matching → semantic memory → primary
reasoning. Any strong local result returns before the primary, fast, or embedding
provider is called. Semantic results are text recommendations, never executable
plans. Confirmation rules are unchanged.

`core/memory/semantic_index.py` adds a lazy SQLite BLOB cache at
`memory/semantic_cache.db`. It stores model, typed namespace, item ID, SHA-256
content hash, dimension, float32 vector, and timestamp; **no raw document or query
text is cached**. Learned knowledge remains in its existing table. A separate
`learned_solution` namespace preserves room for future explicit personal-memory
adapters (preferences, people, goals, projects, and summaries) without merging
their source stores or automatically indexing private profile data.

Only verified/trusted, filtered learned solutions enter retrieval. Up to 200
recent eligible records are considered per lookup; embeddings are batched in
groups of 16. Cosine matching requires at least 0.80 similarity, then reranks
using stored confidence, verification state, success count, recent updates,
explicit teaching/corrections, and an importance value (0.5 until a source adapter
provides one). Recent explicit corrections outrank otherwise equal older matches.
Candidates and rejected knowledge cannot answer through semantics.

Content hashes invalidate changed/corrected document vectors. Different model
names use separate caches. Query-vector retention is bounded to 100 entries.
Cache connections close after each lookup. Clear the ignored cache manually if a
model is replaced with different weights under the same name. No full vector
database, neural search library, or new runtime dependency is required.

Semantic retrieval is optional and lazy. Disabled semantics, unsupported embedding
providers, absent models, invalid vectors, unavailable Ollama, and cache failures
fall through to primary reasoning or the existing offline-unknown response.
The embedding timeout is bounded at 45 seconds to allow cold loading; this wait
occurs only after deterministic paths fail. First indexing is slower than reuse.

Learning still defaults to `ALPHA_AUTO_LEARN_OLLAMA=false`. The existing pipeline
filters before extraction/deduplication, stores lower-confidence candidates only
when enabled or explicitly requested, and verifies only through feedback.
Long usable replies are deterministically distilled into up to four complete
leading sentences/lines and at most 800 characters. Oversized raw output,
terminal-sized dumps, and sensitive output are rejected. Feedback compares the
distilled answer to saved content, so compaction cannot verify a conflicting answer.

Run the non-destructive manual check:

```
python scripts/ollama_smoke_test.py
```

It checks all roles, sends one tiny generation request (thinking disabled for that
test only), sends a harmless embedding request, and measures local routes and
first/cached semantic retrieval with a temporary synthetic memory store. It does
not print model answers, vectors, credentials, or personal memory, and performs
no Windows action. Cold-load latency depends on hardware/model residency; timing
reports are measurements, not test assertions.

API references: [Ollama embeddings](https://docs.ollama.com/api/embed) and
[local model listing](https://docs.ollama.com/api/tags).

No changes to `ui.py`, talking face, layout, controls, shortcuts, visible branding,
or frontend framework are required for this integration.

## Validation and Phase 3

Run `python -m unittest discover -s tests -v` or `python -m pytest`, then
`python -m compileall -q .` and `git diff --check`. Tests use temporary databases
and mocked HTTP and require no Ollama service or GUI dependencies.

Next steps: offline voice, candidate/conflict review behind the existing UI, structured
outcome adapters for more actions, per-session/user context isolation, indexed
token search and bounded retention, and Windows end-to-end action tests. Confirm
learning quality before extending automation. Live Gemini/Deepgram paths and the
existing UI remain available. No model dumps, self-modifying
code, automatic model downloads, or unrestricted autonomy are included here.
