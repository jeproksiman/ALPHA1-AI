"""Small local payload objects used by existing action helpers, without an AI SDK."""
from types import SimpleNamespace


class Part(SimpleNamespace):
    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return cls(inline_data=SimpleNamespace(data=data, mime_type=mime_type))


class GenerateContentConfig(SimpleNamespace):
    pass
