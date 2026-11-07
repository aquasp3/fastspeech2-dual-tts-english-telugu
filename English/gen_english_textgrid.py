"""Pick a ~6.2 h Svarah subset (seed 1234, 1-11 s), then build phone TextGrids from
MMS_FA forced WORD alignment (phone boundaries inside a word are estimated:
vowels weighted 1.5, consonants 1.0). Failed utterances are dropped; no proportional fallback."""
import argparse, random, re, shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tgt
import torch
import torchaudio

from text.english_text import is_vowel, word_to_phones
from text.symbols import symbols

MMS_SR = 16000
SYM = set(symbols)


def clean_words(text):
    t = text.lower()
    if re.search(r"[0-9]", t):
        raise ValueError("digits in transcript")
    t = re.sub(r"[^a-z' ]", " ", t)
    words = [w.strip("'") for w in t.split()]
    words = [w for w in words if w]
    if not words:
        raise ValueError("empty transcript")
    return words


def pick_subset(src, hours, seed):
    items = []
    for wav in sorted((src / "speaker1").glob("*.wav")):
        lab = src / "speaker1" / (wav.stem + ".lab")
        if not lab.exists():
            continue
        d = sf.info(str(wav)).duration
        if 0.5 <= d <= 18.0:
            items.append((wav, lab, d))
    random.Random(seed).shuffle(items)
    chosen, total = [], 0.0
    for it in items:
        if total >= hours * 3600:
            break
        chosen.append(it)
        total += it[2]
    return chosen, total


def align(model, tokenizer, aligner, wav_path, words, device):
    y, _ = librosa.load(str(wav_path), sr=MMS_SR, mono=True)
    wav = torch.from_numpy(y).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = model(wav)
    token_spans = aligner(emission[0].cpu(), tokenizer(words))
    if len(token_spans) != len(words):
        raise ValueError("word/span count mismatch")
    ratio = wav.shape[1] / emission.shape[1] / MMS_SR
    spans = [(ts[0].start * ratio, ts[-1].end * ratio) for ts in token_spans]
    return spans, wav.shape[1] / MMS_SR


def write_textgrid(path, spans, words, duration):
    ivs, cur = [], 0.0
    for (s, e), w in zip(spans, words):
        s = max(s, cur)
        e = min(e, duration)
        if e <= s:
            raise ValueError("empty word span for %r" % w)
        phones = word_to_phones(w)
        if any("@" + p not in SYM for p in phones):
            raise ValueError("phone not in symbol set for %r: %s" % (w, phones))
        if s > cur:
            ivs.append((cur, s, "sp"))
        wts = np.array([1.5 if is_vowel(p) else 1.0 for p in phones])
        edges = list(s + np.concatenate(([0.0], np.cumsum(wts / wts.sum() * (e - s)))))
        edges[0], edges[-1] = s, e
        ivs += [(edges[i], edges[i + 1], p) for i, p in enumerate(phones)]
        cur = e
    if cur < duration:
        ivs.append((cur, duration, "sp"))
    tier = tgt.core.IntervalTier(0.0, duration, name="phones")
    for a, b, p in ivs:
        if b > a:
            tier.add_interval(tgt.core.Interval(float(a), float(b), p))
    tg = tgt.core.TextGrid()
    tg.add_tier(tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    tgt.io.write_to_file(tg, str(path), format="long")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../raw_data/Svarah")
    ap.add_argument("--dst", default="../../raw_data/Svarah_sub")
    ap.add_argument("--out", default="preprocessed_data/Svarah")
    ap.add_argument("--hours", type=float, default=6.2)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    src, dst, out = Path(a.src), Path(a.dst), Path(a.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device)
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    chosen, total = pick_subset(src, a.hours, a.seed)
    print("Selected %d utterances, %.2f h (before drops)" % (len(chosen), total / 3600))
    kept, kept_sec, dropped = 0, 0.0, 0
    for i, (wav, lab, dur) in enumerate(chosen):
        try:
            words = clean_words(lab.read_text(encoding="utf-8"))
            spans, d = align(model, tokenizer, aligner, wav, words, device)
            write_textgrid(out / "TextGrid" / "speaker1" / (wav.stem + ".TextGrid"), spans, words, dur)
            (dst / "speaker1").mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, dst / "speaker1" / wav.name)
            (dst / "speaker1" / (wav.stem + ".lab")).write_text(" ".join(words), encoding="utf-8")
            kept += 1
            kept_sec += dur
        except Exception as exc:
            dropped += 1
            print("DROP %s: %s" % (wav.name, exc))
        if (i + 1) % 200 == 0:
            print("%d/%d done, kept %d" % (i + 1, len(chosen), kept))
    print("KEPT %d utterances, %.2f h; DROPPED %d" % (kept, kept_sec / 3600, dropped))


if __name__ == "__main__":
    main()
