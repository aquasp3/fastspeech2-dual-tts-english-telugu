import argparse, os, random
import numpy as np, librosa, soundfile as sf, tgt
from text.telugu_text import text_to_phones


def weight(p):
    if p == "sp":
        return 0.6
    c = ord(p)
    if c == 0x0C4D:
        return 0.25
    if 0x0C3E <= c <= 0x0C4C or c in (0x0C55, 0x0C56):
        return 1.2
    if c in (0x0C01, 0x0C02, 0x0C03):
        return 0.7
    return 1.0


def allocate(weights, total):
    w = np.array(weights, float)
    raw = w / w.sum() * (total - len(w))
    base = np.floor(raw).astype(int)
    rem = (total - len(w)) - base.sum()
    base[np.argsort(-(raw - base))[:rem]] += 1
    return base + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../raw_data/Telugu")
    ap.add_argument("--out", default="preprocessed_data/Telugu")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--max_sec", type=float, default=11.0)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--top_db", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rows = []
    with open(os.path.join(a.raw, "train.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) >= 3:
                rows.append((p[0], p[1], p[2].strip()))
    random.Random(a.seed).shuffle(rows)

    total, budget, kept = 0.0, a.hours * 3600, []
    t = lambda k: float(k) * a.hop / a.sr
    for base, spk, txt in rows:
        if total >= budget:
            break
        wav = os.path.join(a.raw, spk, base + ".wav")
        if not os.path.isfile(wav):
            continue
        dur = sf.info(wav).duration
        if not (a.min_sec <= dur <= a.max_sec):
            continue
        phones = text_to_phones(txt)
        if len(phones) < 3:
            continue
        y, _ = librosa.load(wav, sr=a.sr)
        _, (i0, i1) = librosa.effects.trim(y, top_db=a.top_db)
        f0 = int(round(i0 / a.hop))
        f1 = min(int(round(i1 / a.hop)), len(y) // a.hop)
        if f1 - f0 < len(phones):
            continue
        frames = allocate([weight(p) for p in phones], f1 - f0)
        bounds = f0 + np.concatenate([[0], np.cumsum(frames)])
        dur_s = len(y) / a.sr

        tier = tgt.core.IntervalTier(start_time=0.0, end_time=dur_s, name="phones")
        if f0 > 0:
            tier.add_interval(tgt.core.Interval(0.0, t(f0), "sil"))
        for p, s, e in zip(phones, bounds[:-1], bounds[1:]):
            tier.add_interval(tgt.core.Interval(t(s), t(e), p))
        if dur_s > t(f1):
            tier.add_interval(tgt.core.Interval(t(f1), dur_s, "sil"))

        tg = tgt.core.TextGrid()
        tg.add_tier(tier)
        od = os.path.join(a.out, "TextGrid", spk)
        os.makedirs(od, exist_ok=True)
        tgt.io.write_to_file(tg, os.path.join(od, base + ".TextGrid"), format="long")
        with open(os.path.join(a.raw, spk, base + ".lab"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        total += dur
        kept.append("{}|{}|{:.2f}".format(base, spk, dur))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "selected_utterances.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    print("Selected {} utterances, {:.2f} hours".format(len(kept), total / 3600))


if __name__ == "__main__":
    main()