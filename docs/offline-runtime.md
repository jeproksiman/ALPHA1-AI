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

Offline microphone support is optional. Install Vosk and its model separately,
then configure an existing local directory:

```powershell
$env:ALPHA_OFFLINE_STT_MODEL_PATH = 'C:\models\vosk-model-small-en-us-0.15'
$env:ALPHA_OFFLINE_TTS = 'true'
```

This pipeline uses local Vosk recognition and, when enabled and installed,
Windows SAPI via pywin32. It never downloads assets or falls back to cloud speech.
Mute, push-to-talk and interrupt retain their existing controls; TTS uses the
existing SPEAKING face state. Without a configured model/device/dependency, a
single status message explains that typed offline input remains available.
The existing Whisper/Kokoro engines are not selected automatically because
their initialization can download models. Optional packages are not installed
by runtime startup. `.env.example` documents settings; export variables or use
private configuration (the example file is not loaded automatically).

Manual checks on the real desktop:

1. With internet on and a valid provider key, use AUTO and launch `python main.py`.
   Confirm the normal live connection and microphone work.
2. Quit, switch Wi-Fi off, and launch again. Expect OFFLINE status, no cloud
   traceback/retry storm. Type `open calculator`, `what do you remember about me?`,
   and `explain what ModuleNotFoundError means`. Verify the local action, personal
   memory response and local Ollama reasoning. Do not enable airplane mode if
   another connection is needed for remote access.
3. Start online, then disconnect Wi-Fi while running. The live session should
   close and offline input should keep working; a stalled session is detected
   within the conservative 45-second monitor interval plus probe timeout.
4. Reconnect Wi-Fi. AUTO checks quietly, waits for an active local response or
   speech to finish, and restores live support. Check the microphone after recovery.
5. Set mode `offline` and repeat with Ollama stopped. A deterministic command and
   stored memory should still work. Restore the environment/model service afterward.

Automated tests mock cloud transport and providers; they do not turn off Wi-Fi
or validate physical microphone quality. `python scripts/ollama_smoke_test.py`
checks installed local models using temporary synthetic memory.
