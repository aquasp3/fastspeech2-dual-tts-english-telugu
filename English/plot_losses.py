import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "../Results/English/log"
OUT = "../Results/English"
pat = re.compile(r"Step\s+(\d+)(?:/\d+)?,\s*Total Loss:\s*([-\d.eE+]+)")


def parse(path):
    d = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    s = sorted(d)
    return s, [d[k] for k in s]


for split, fname, title in [("train", "training_loss.png", "English Training Loss"),
                            ("val", "validation_loss.png", "English Validation Loss")]:
    s, v = parse(os.path.join(LOG, split, "log.txt"))
    print(split, len(s), "points; last:", (s[-1], v[-1]) if s else None)
    if split == "val" and s:
        i = v.index(min(v))
        print("val minimum %.4f at step %d" % (v[i], s[i]))
    plt.figure(figsize=(8, 4.5))
    plt.plot(s, v, marker="o" if split == "val" else None, ms=3)
    plt.xlabel("Step")
    plt.ylabel("Total loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close()
