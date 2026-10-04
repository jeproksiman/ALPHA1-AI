import re

_SECRET = re.compile(
    r"AIza[\w-]{35}|gh[pousr]_[\w]{20,}|github_pat_[\w]{20,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|sk-[\w-]{16,}|"
    r"-----BEGIN [\w ]*PRIVATE KEY-----|"
    r"\b(?:api[_ -]?key|password|passwd|token|secret|authorization|"
    r"client[_ -]?secret|private[_ -]?key|credentials?|cookies?|"
    r"session[_ -]?(?:id|cookie)|authentication[_ -]?cookies?|auth[_ -]?cookies?|"
    r"(?:access|refresh|auth|id)[_ -]?token|secret[_ -]?key|api[_ -]?token|access[_ -]?key)\b\s*[\"']?\s*[:=]|"
    r"\b(?:my\s+)?(?:password|api[_ -]?key|token|private[_ -]?key|credentials?)\s+is\s+\S+|"
    r"\bBearer\s+\S+|eyJ[\w-]+\.[\w-]+\.[\w-]+", re.I)
_DANGEROUS = re.compile(
    r"\b(?:remove-item|del|erase|rmdir|rm|format|format-volume|diskpart|"
    r"clear-disk|reg delete|remove-itemproperty|invoke-expression|iex|"
    r"set-mppreference|add-mppreference|disable-security|mimikatz)\b|"
    r"(?:delete|disable|extract|dump|wipe|format)\s+.{0,35}"
    r"(?:files?|disk|registry|security|defender|credentials?|passwords?|tokens?)|"
    r"powershell.{0,40}(?:-enc|encodedcommand)|\b(?:curl|iwr).{0,100}\|", re.I)


def contains_secret(text):
    return bool(_SECRET.search(str(text)))


def unsafe_to_learn(text):
    return contains_secret(text) or bool(_DANGEROUS.search(str(text)))


class KnowledgeExtractor:
    """Reject oversized/sensitive output rather than truncate a secret away."""
    def extract(self, user_input, response):
        if not isinstance(response, str) or not isinstance(user_input, str):
            return None
        if unsafe_to_learn(user_input + "\n" + response):
            return None
        if len(response) > 2000 or len(user_input) > 500:
            return None
        response = response.strip()
        if len(response) < 12 or response.count("\n") > 25:
            return None
        if len(response)>800:
            # Retain whole leading sentences/lines, never an arbitrary truncated dump.
            units = re.split(r'(?<=[.!?])\s+|\n+',response)
            concise = []
            for unit in units:
                if sum(len(value)+1 for value in concise)+len(unit)>800:
                    break
                concise.append(unit.strip())
                if len(concise)>=4:
                    break
            response = ' '.join(concise)
            if len(response)<12:
                return None
        return {"topic": " ".join(user_input.split())[:100],
                "trigger_text": user_input.strip(), "response": response,
                "solution": response}
