import re
from functools import lru_cache

from g2p_en import G2p

_VALID_PHONE_RE = re.compile(r"^[A-Z]+[0-2]?$")
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


@lru_cache(maxsize=1)
def _g2p():
    return G2p()


def normalize_for_g2p(text):
    text = text.lower().strip()
    if not re.fullmatch(r"[a-z']+(?: [a-z']+)*", text):
        raise ValueError("Transcript contains unsupported characters")
    return text


def _valid_phone(phone):
    return bool(_VALID_PHONE_RE.fullmatch(phone))


def word_to_phones(word):
    phones = [p for p in _g2p()(word) if p != " " and _valid_phone(p)]
    if not phones:
        raise ValueError("g2p-en produced no valid phones for %r" % word)
    return phones


def text_to_phones(text):
    words = normalize_for_g2p(text).split()
    phones = []
    for word in words:
        phones.extend(word_to_phones(word))
        phones.append("sp")
    if phones:
        phones.pop()
    return phones


def is_vowel(phone):
    return phone.rstrip("012") in _VOWELS

