import argparse, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from text import text_to_sequence
from text.english_text import word_to_phones
from text.symbols import symbols
from utils.model import get_model, get_vocoder, vocoder_infer

SYM = set(symbols)
SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def below100(n):
    return ONES[n] if n < 20 else TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])


def cardinal(n):
    if n < 100:
        return below100(n)
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + below100(n % 100))
    if n < 1000000:
        return cardinal(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + cardinal(n % 1000))
    return " ".join(ONES[int(d)] for d in str(n))


def year(n):
    if 2000 <= n < 2010:
        return "two thousand" + ("" if n == 2000 else " " + ONES[n - 2000])
    hi, lo = n // 100, n % 100
    tail = "hundred" if lo == 0 else ("oh " + ONES[lo] if lo < 10 else below100(lo))
    return below100(hi) + " " + tail


def _time(m):
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    out = [below100(h)]
    if mi:
        out.append("oh " + ONES[mi] if mi < 10 else below100(mi))
    if ap:
        out.append("ay em" if ap.lower() == "am" else "pee em")
    return " " + " ".join(out) + " "


def normalize(text):
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", _time, text, flags=re.I)
    t = re.sub(r"\b(\d{4})\b", lambda m: " " + year(int(m.group(1))) + " ", t)
    t = re.sub(r"\b(\d+)\b", lambda m: " " + cardinal(int(m.group(1))) + " ", t)
    t = re.sub(r"[^a-z' ]", " ", t.lower())
    return [w.strip("'") for w in t.split() if w.strip("'")]


def build(text):
    words = normalize(text)
    seq = []
    for i, w in enumerate(words):
        if i:
            seq.append("sp")
        for p in word_to_phones(w):
            if "@" + p in SYM:
                seq.append(p)
            else:
                print("  skipping unknown phone", p, "in", w)
    ids = np.array(text_to_sequence("{" + " ".join(seq) + "}", ["english_cleaners"]))
    assert len(ids) == len(seq), "phone/id length mismatch"
    return words, seq, ids


def plot(path, labels, values, title, ylabel, bars):
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 4))
    x = np.arange(len(labels))
    if bars:
        ax.bar(x, values)
    else:
        ax.plot(x, values, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlabel("Phone")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("-p", default="config/preprocess.yaml")
    ap.add_argument("-m", default="config/config.yaml")
    ap.add_argument("-t", default="config/train.yaml")
    ap.add_argument("--out", default="../Results/English")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc = yaml.load(open(args.p), Loader=yaml.FullLoader)
    mc = yaml.load(open(args.m), Loader=yaml.FullLoader)
    tc = yaml.load(open(args.t), Loader=yaml.FullLoader)
    model = get_model(args, (pc, mc, tc), device, train=False)
    vocoder = get_vocoder(mc, device)
    stats = json.load(open(os.path.join(pc["path"]["preprocessed_path"], "stats.json")))
    p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
    e_mean, e_std = stats["energy"][2], stats["energy"][3]
    hop = pc["preprocessing"]["stft"]["hop_length"]
    sr = pc["preprocessing"]["audio"]["sampling_rate"]
    for d in ("test_generated_audios", "variance_adaptor_plots/duration",
              "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    for sid, text in SENTENCES.items():
        words, phones, ids = build(text)
        print(sid, "|", " ".join(words))
        print("  phones:", " ".join(phones))
        n = len(ids)
        texts = torch.from_numpy(ids).long().unsqueeze(0).to(device)
        lens = torch.tensor([n]).to(device)
        speakers = torch.tensor([0]).long().to(device)
        with torch.no_grad():
            out = model(speakers, texts, lens, n)
            mel_len = out[9]
            wav = vocoder_infer(out[1].transpose(1, 2), vocoder, mc, pc, lengths=mel_len * hop)[0]
        wavfile.write(os.path.join(args.out, "test_generated_audios", sid + ".wav"), sr, wav)
        dur = out[5][0, :n].cpu().numpy()
        pitch = out[2][0, :n].cpu().numpy() * p_std + p_mean
        energy = out[3][0, :n].cpu().numpy() * e_std + e_mean
        tag = "%s (checkpoint %d)" % (sid, args.restore_step)
        base = os.path.join(args.out, "variance_adaptor_plots")
        plot(os.path.join(base, "duration", sid + ".png"), phones, dur,
             "Predicted duration - " + tag + ", total %.2f s" % (dur.sum() * hop / sr), "Frames", True)
        plot(os.path.join(base, "pitch", sid + ".png"), phones, pitch,
             "Predicted pitch (F0) - " + tag, "Hz", False)
        plot(os.path.join(base, "energy", sid + ".png"), phones, energy,
             "Predicted energy - " + tag, "Energy", False)
    print("Done. Outputs in", args.out)


if __name__ == "__main__":
    main()
