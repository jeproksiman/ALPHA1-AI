JARVIS now starts with Ollama as its only AI brain. `python main.py` opens the
existing UI and local microphone without a Gemini, OpenAI, Deepgram, or Anthropic
key. Normal startup never probes or opens a cloud AI session, including when an
old private configuration still selects AUTO/LIVE or contains cloud credentials.

Export environment variables in PowerShell (or use the corresponding lowercase
`alpha_*` keys in the private `config/api_keys.json`). `.env.example` documents
settings; it is not automatically loaded:

```powershell
$env:ALPHA_BRAIN_PROVIDER = 'ollama'
$env:ALPHA_CLOUD_AI_ENABLED = 'false'
$env:ALPHA_WEB_TOOLS_ENABLED = 'true'
python main.py
```

The brain provider is deliberately pinned to Ollama. The cloud flag guards
retained legacy helpers; it does not activate a cloud session or change normal
reasoning. `live:` also stays local. Existing provider settings remain on disk.

Registered known commands run directly. Simple action selection uses
`qwen3.5:0.8b`; complex planning, conversation, and final responses use
`gemma4:e4b`. `nomic-embed-text:latest` supplies semantic retrieval. Structured
plans use the same discovered action/plugin registry and validate names, required
arguments, types, enums, and unexpected arguments before dispatch. Dangerous or
unclassified actions require the existing HUD confirmation. Complex requests
have at most three executed steps, reject repeated steps, and stop at a pending
confirmation. There is no model-generated shell executor. Desktop natural
language selects supported deterministic operations; unsupported operations are
reported honestly. Dev-agent execution accepts only a saved Python/JavaScript
entry inside its confirmed project, with no arbitrary run command.

Current-info requests use the existing `web_search` retrieval action before
Ollama summarizes. Search uses ordinary web retrieval, not an AI API. Weather
retains the existing action: it opens the requested city's weather search; the
assistant does not claim it fetched a numerical forecast if the tool did not.
`ALPHA_WEB_TOOLS_ENABLED=false` disables identified web actions; network failure
does not disable local commands or memory. Plugins retain their existing
configuration and require confirmation when selected by the model. Network
operations internal to third-party plugins remain the plugin's responsibility.

File uploads, screen/camera analysis, PC controls, reminders, messaging, browser
tools, routines, dashboard text commands, and discovered plugins keep their
existing handlers and UI hooks. Legacy generation helpers route to loopback
Ollama, including images for a locally installed vision-capable primary model.
Missing local model/media capability reports unavailability. Dashboard text and
authenticated phone PCM use the same local brain/STT and memory. Phone PCM is
bounded and respects the local mute/PTT gate; desktop frames pause while phone
frames arrive so the recognizer does not mix both microphones.
System alerts remain local. `ALPHA_PROACTIVE_ENABLED=true` enables a bounded
local check-in after idle time; proactive generation cannot execute tools.

Offline speech now defaults to faster-whisper. Install dependencies with
`python -m pip install -r requirements.txt`. No speech model is downloaded by
startup. Configure a converted model directory or reuse the Hugging Face cache:

```powershell
$env:ALPHA_STT_PROVIDER = 'faster-whisper'
$env:ALPHA_WHISPER_MODEL = 'small' # model size or existing local model directory
$env:ALPHA_WHISPER_DEVICE = 'auto'
$env:ALPHA_WHISPER_COMPUTE_TYPE = 'auto'
$env:ALPHA_ALLOW_MODEL_DOWNLOAD = 'false'
$env:ALPHA_TTS_PROVIDER = 'auto'
$env:ALPHA_PIPER_VOICE_PATH = '' # optional existing .onnx with adjacent .onnx.json
$env:ALPHA_TTS_ENABLED = 'true'
$env:ALPHA_OFFLINE_WARMUP = 'true'
```

If the configured Whisper size is not cached, the smallest complete installed
Whisper snapshot is reused. Otherwise a concise setup message explains missing
assets; Vosk remains a fallback through `ALPHA_OFFLINE_STT_MODEL_PATH`. Explicit
`ALPHA_ALLOW_MODEL_DOWNLOAD=true` permits Whisper's own download only. Piper
voices are never downloaded. AUTO Whisper device uses CPU/int8 with two threads
so Ollama can retain GPU capacity; choose `cuda` explicitly if desired.

Listening uses a bounded in-memory queue, an amplitude/silence detector, and
utterances capped at 12 seconds. No raw microphone recordings are saved.
Whisper loads off the UI thread, releases its model after five idle minutes, and
reloads when speech returns. Optional warmup performs Whisper silence inference,
then sequential Ollama reasoning/embedding warmup after five idle seconds. User
speech/interrupt cancels warmup cooperatively. Generation/embedding keep-alive
limits are two minutes/one minute; the fast model is not warmed unnecessarily.

TTS prefers a configured local Piper voice and falls back to Windows SAPI.
Speech is synthesized as memory-only PCM and played in small interruptible
blocks. The existing live PCM analysis, transcript visemes, HUD schedule and
SPEAKING state drive the same face. ESC/INTERRUPT cancels queued speech and
pending local responses, purges synthesis/playback, and clears pending face
frames. An inference already executing in native Whisper code may finish, but
its cancelled result is discarded. Ollama generation is streamed for cooperative
cancellation; a server blocked before its first token remains bounded by the
request timeout. No overlapping TTS queue is maintained.

`ALPHA_OFFLINE_BARGE_IN=true` enables conservative amplitude barge-in. This is
opt-in because speaker echo depends on the room; headphones are recommended.
Mute, push-to-talk and interrupt use the existing controls. Without local STT,
typed replies can still speak through available TTS. Missing TTS leaves the
same response visible as text. No cloud STT/TTS fallback is used.

Conversation continuity extends the existing SQLite `interaction_history`
table with mode, topic, project, importance, timestamps and summary linkage.
Schema version 3 adds one current context row and a bounded personal-fact store.
Live Gemini/Deepgram transcripts and typed/voice OfflineBrain replies feed the
same store. Recent topic, project, next goal and identity restore on startup.
Conversation turns are not promoted into verified learned actions/solutions.
Secrets are rejected before persistence or model context. Existing personal
JSON preferences remain available through a bounded sanitized profile adapter.
Live providers receive at most 1,800 characters of shared local context; Ollama
receives compact recent context plus at most two relevant verified solutions.
The entire database is never injected. Session summarization now stays local.
No automatic spoken greeting was added; a restored topic appears in status text,
so disabled startup speech remains respected.

Offline personal commands include `call me NAME`, `I prefer ...`, `I am working
on ...`, `remember that ...`, `what do you remember about me?`, `what were we
just talking about?`, `what were we testing?`, `what am I working on?`, and
`what did I tell you earlier?`. `remember this` retains the previous user message
and preserves existing solution-learning feedback. `forget that` deletes the
latest conversation and its saved note/goal. It does not wipe unrelated memory.

Real desktop checks (these verify physical audio and visible face motion):

1. With Wi-Fi OFF, export mode `offline` and run `python main.py`. Wait for local
   STT/microphone readiness. Say **Hey, can you hear me?** Verify transcript,
   local answer, audible speech and the existing moving mouth. Check a typed
   reply speaks too. Try calculator and stored personal memory with Ollama stopped.
2. Say **Remember that we are testing offline voice.** Close and restart offline.
   Ask **What were we testing?** Verify saved topic without Gemini.
3. With internet available, search for current news and verify retrieved results
   are summarized locally. Disconnect Wi-Fi and check a known local command and
   topic recall. Restore Wi-Fi and repeat search. No cloud AI session is opened.
4. Press ESC/INTERRUPT during a long reply and during reasoning. Verify audio and
   face stop, no stale reply is spoken, and a new request can be answered.
5. Test mute/PTT and optionally barge-in with headphones. Return environment
   variables to your preferred settings afterward.

`python scripts/offline_voice_smoke_test.py` tests cached Whisper with synthetic
Windows speech in memory, without physical recording/playback. Existing
`python scripts/ollama_smoke_test.py` checks local model roles. Automated tests
mock heavy models and devices; they do not certify microphone quality or visible
rendering on your physical display. `.env.example` documents settings; export
variables or use private configuration (the example is not loaded automatically).

Provider implementation references: [faster-whisper local model loading](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
and [Piper local synthesis](https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/voice.py).

Local voice configuration (no UI redesign):

```powershell
$env:ALPHA_TTS_PROVIDER = 'sapi' # auto, piper, kokoro, sapi
$env:ALPHA_TTS_VOICE = 'David'   # installed SAPI name; leave blank for default
$env:ALPHA_TTS_RATE = '1.1'      # speed multiplier, bounded 0.5..2
$env:ALPHA_TTS_PITCH = '-2'     # SAPI pitch only, bounded -10..10
$env:ALPHA_TTS_VOLUME = '0.8'   # PCM volume, bounded 0..1
```

List Windows voices using PowerShell:
`(New-Object -ComObject SAPI.SpVoice).GetVoices() | ForEach-Object { $_.GetDescription() }`.
For Piper, install `piper-tts`, set `ALPHA_PIPER_VOICE_PATH` to an existing
`.onnx` with its adjacent `.onnx.json`, and choose a speaker name/id with
`ALPHA_TTS_VOICE` for multi-speaker models. For Kokoro, install `kokoro-onnx`,
set `ALPHA_KOKORO_MODEL_PATH` and `ALPHA_KOKORO_VOICES_PATH` to local model and
voice-bank files, and choose a voice such as `af_heart` or `bm_george`.
Auto prefers configured Piper, configured Kokoro, then Windows SAPI. Missing
optional engines fall back to SAPI with a status message. Neither neural
provider downloads assets. Piper/Kokoro use native pitch; SAPI supports the
pitch setting. All engines feed the same interruptible PCM playback and
existing face analyser. Voice timbre and microphone accuracy still need a
physical desktop listening test.

Implementation references: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[Piper synthesis settings](https://github.com/OHF-Voice/piper1-gpl/blob/main/src/piper/config.py),
and [local Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/save.py).
