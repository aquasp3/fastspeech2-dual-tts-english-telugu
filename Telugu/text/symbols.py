import unicodedata

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@spn", "@sil"]
_telugu = ["@" + chr(c) for c in range(0x0C00, 0x0C80) if unicodedata.name(chr(c), None)]

symbols = [_pad] + list(_special) + list(_punctuation) + _telugu + _silences