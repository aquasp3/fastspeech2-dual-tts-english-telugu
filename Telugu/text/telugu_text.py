import re
import unicodedata

_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది",
          "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను",
          "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై",
         7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below100(n):
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + (" " + _UNITS[u] if u else "")


def _below1000(n):
    h, r = divmod(n, 100)
    out = []
    if h:
        if h == 1:
            out.append("నూట" if r else "వంద")
        else:
            out.append(_UNITS[h] + (" వందల" if r else " వందలు"))
    if r:
        out.append(_below100(r))
    return " ".join(out)


def num_to_telugu(n):
    if n < 1000:
        return _below1000(n) if n else _UNITS[0]
    if n < 100000:
        t, r = divmod(n, 1000)
        head = "వెయ్యి" if t == 1 else _below100(t) + (" వేల" if r else " వేలు")
        return head + (" " + _below1000(r) if r else "")
    return " ".join(_UNITS[int(d)] for d in str(n))


def _ordinal(words):
    return words[:-1] + "వ" if words[-1] in "ుి" else words + "వ"


def _time(m):
    h, mi = int(m.group(1)), int(m.group(2))
    return num_to_telugu(h) + (" " + num_to_telugu(mi) if mi else "")


def normalize_telugu(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"(\d{1,2}):(\d{2})", _time, text)
    text = re.sub(r"(\d+)వ(?![\u0C00-\u0C7F])",
                  lambda m: _ordinal(num_to_telugu(int(m.group(1)))), text)
    text = re.sub(r"\d+", lambda m: num_to_telugu(int(m.group())), text)
    return text


_VALID = {chr(c) for c in range(0x0C00, 0x0C80) if unicodedata.name(chr(c), None)}


def text_to_phones(text):
    phones = []
    for ch in normalize_telugu(text):
        if ch in _VALID:
            phones.append(ch)
        elif ch.isspace() or ch in ",.;:!?-()'\"।":
            if phones and phones[-1] != "sp":
                phones.append("sp")
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones