"""Deepgram Voice Agent transport and settings conversion."""
from __future__ import annotations

import asyncio
import json


def _json_schema(value, key: str = ""):
    if isinstance(value, dict):
        return {name: _json_schema(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_json_schema(item) for item in value]
    if key == "type" and isinstance(value, str):
        return value.lower()
    return value


def build_agent_settings(system_prompt: str, tool_declarations: list[dict]) -> dict:
    functions = []
    for declaration in tool_declarations:
        if declaration.get("name") == "screen_process":
            continue
        function = {
            "name": declaration["name"],
            "description": declaration.get("description", ""),
            "parameters": _json_schema(
                declaration.get("parameters", {"type": "OBJECT", "properties": {}})
            ),
            "defer_until_eot": True,
        }
        functions.append(function)

    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 16000},
            "output": {
                "encoding": "linear16",
                "sample_rate": 24000,
                "container": "none",
            },
        },
        "agent": {
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-3",
                    "language": "multi",
                }
            },
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "prompt": system_prompt,
                "functions": functions,
            },
            "speak": {
                "provider": {"type": "deepgram", "model": "aura-2-thalia-en"}
            },
        },
    }


class DeepgramAgent:
    """Async WebSocket client for the Deepgram Voice Agent API."""

    URL = "wss://agent.deepgram.com/v1/agent/converse"

    def __init__(self, api_key: str, settings: dict):
        self.api_key = api_key
        self.settings = settings
        self._socket = None

    async def __aenter__(self):
        from websockets.asyncio.client import connect

        self._socket = await connect(
            self.URL,
            additional_headers={"Authorization": f"Token {self.api_key}"},
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        )
        try:
            welcome = await asyncio.wait_for(self.receive(), timeout=15)
            if not isinstance(welcome, dict) or welcome.get("type") != "Welcome":
                raise RuntimeError(f"Deepgram did not welcome the session: {welcome}")

            await self._socket.send(json.dumps(self.settings))
            while True:
                response = await asyncio.wait_for(self.receive(), timeout=30)
                if isinstance(response, dict) and response.get("type") == "SettingsApplied":
                    return self
                if isinstance(response, dict) and response.get("type") == "Error":
                    code = response.get("code", "Error")
                    description = response.get("description") or str(response)
                    raise RuntimeError(f"Deepgram {code}: {description}")
        except BaseException:
            await self._socket.close()
            self._socket = None
            raise

    async def __aexit__(self, *_exc):
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def send_audio(self, audio: bytes) -> None:
        await self._socket.send(audio)

    async def send_realtime_input(self, audio) -> None:
        await self.send_audio(audio.data)

    async def send_text(self, text: str) -> None:
        await self._socket.send(json.dumps({"type": "InjectUserMessage", "content": text}))

    async def send_client_content(self, turns, turn_complete: bool = True) -> None:
        del turn_complete
        if isinstance(turns, dict):
            turns = [turns]
        text = "\n".join(
            part.get("text", "")
            for turn in (turns or [])
            for part in (turn.get("parts", []) if isinstance(turn, dict) else [])
            if isinstance(part, dict)
        ).strip()
        if text:
            await self.send_text(text)

    async def send_function_result(self, call_id: str, name: str, content: str) -> None:
        await self._socket.send(json.dumps({
            "type": "FunctionCallResponse",
            "id": call_id,
            "name": name,
            "content": content,
        }))

    async def send_tool_response(self, function_responses) -> None:
        for response in function_responses:
            content = getattr(response, "response", "")
            if isinstance(content, dict):
                content = content.get("result", content)
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            await self.send_function_result(
                str(getattr(response, "id", "")),
                str(getattr(response, "name", "")),
                content,
            )

    async def receive(self):
        message = await self._socket.recv()
        return json.loads(message) if isinstance(message, str) else message