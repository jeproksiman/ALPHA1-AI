"""Text matching only; this module never executes a command."""
import re
import unicodedata
from difflib import SequenceMatcher


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def similarity(left, right):
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    a, b = set(left.split()), set(right.split())
    ratio = SequenceMatcher(None, left, right).ratio()
    return min(0.89, max(0.90 * ratio, 0.55 * ratio + 0.45 * len(a & b) / len(a | b)))
