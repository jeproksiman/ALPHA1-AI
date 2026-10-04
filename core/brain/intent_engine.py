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


def knowledge_similarity(left, right):
    """Small deterministic error vocabulary; no embeddings or action parameter inference."""
    def tokens(text):
        value = normalize(text)
        value = re.sub(r'\b(?:modulenotfounderror|no module named|module not found)\b','module missing',value)
        stop = {'my','the','a','an','is','says','please','how','do','i','fix','error','in'}
        return set(value.split()) - stop
    a,b = tokens(left),tokens(right)
    if not a or not b:
        return similarity(left,right)
    # Never ignore a changed package/application identifier or opposite instruction.
    for opposite in ({'enable','disable'},{'install','uninstall'},{'open','close'}):
        if a & opposite and b & opposite and a & opposite != b & opposite:
            return 0.0
    package_a = {token for token in a if any(char.isdigit() for char in token)}
    package_b = {token for token in b if any(char.isdigit() for char in token)}
    if package_a and package_b and package_a != package_b:
        return 0.0
    overlap = len(a & b)
    keyword = 0.60 * overlap/len(a | b) + 0.40 * overlap/min(len(a),len(b))
    return min(0.95,max(similarity(left,right),keyword))
