JARVIS starts in `auto` runtime mode. The selected Gemini/Deepgram endpoint is
checked with a two-second TLS deadline before any live session is created.
Missing/rejected keys and unavailable transport leave typed local input usable.
Authentication failures are reported separately and are not retried with the
same key in AUTO. Configure a new key using the existing settings control.

Set environment variables in PowerShell before launching (or use the corresponding
lowercase `alpha_*` keys in the private `config/api_keys.json`):

```powershell
$env:ALPHA_RUNTIME_MODE = 'auto' # auto, offline, or live
python main.py
```

`offline` never probes or connects to Gemini/Deepgram. `live` retains the live
provider connection/retry behavior. AUTO checks unavailable transport every
45 seconds, waits for local processing/speech to finish, and reconnects using
the single provider loop. An active AUTO session also checks transport every
45 seconds to catch stalled sockets. Typed requests continue through OfflineBrain
without needing a cloud session. An explicit `live:` prefix uses local processing
while offline. If Ollama is stopped, registered commands and verified SQLite
memory still work; unknown reasoning reports local model unavailability.

Offline speech now defaults to faster-whisper. Install dependencies with
`python -m pip install -r requirements.txt`. No speech model is downloaded by
startup. Configure a converted model directory or reuse the Hugging Face cache:

```powershell
$env:ALPHA_OFFLINE_STT_PROVIDER = 'faster-whisper'
$env:ALPHA_WHISPER_MODEL = 'small' # model size or existing local model directory
$env:ALPHA_WHISPER_DEVICE = 'auto'
$env:ALPHA_WHISPER_COMPUTE_TYPE = 'auto'
$env:ALPHA_ALLOW_MODEL_DOWNLOAD = 'false'
$env:ALPHA_OFFLINE_TTS_PROVIDER = 'auto'
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
3. Start online in AUTO, discuss improving offline voice, then disconnect Wi-Fi.
   Ask **What were we just discussing?** Verify offline recall. Tell ALPHA your
   next goal is wake-word detection; reconnect and verify live continuity after
   idle recovery. Do not disconnect a connection needed for remote access.
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
