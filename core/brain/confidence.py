import math

LOCAL_EXECUTION_THRESHOLD = 0.80
OLLAMA_FALLBACK_THRESHOLD = 0.80


def clamp(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Confidence must be finite")
    return max(0.0, min(1.0, value))


def feedback_confidence(value, success):
    return clamp(value + 0.10 if success else value - 0.25)
