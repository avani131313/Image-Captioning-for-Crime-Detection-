"""Fit one linear probe per check on frozen CLIP embeddings.

    python -m scripts.train_probes --verified data/ucf/verified.jsonl

WHAT A PROBE IS

A check currently works by comparing an image against sentences written by
hand, and the whole decision rests on a cosine margin of about 0.03. A probe
replaces those sentences with a single vector FITTED to real examples:

    score = sigmoid(w . embedding + b)

Same dot product, same speed, one vector per check - just as inspectable.
But optimised for exactly this decision rather than being a sentence that
happens to land near the right place in CLIP space.

SCOPE

UCF frames are WHOLE FRAMES, so these are frame-level probes matching
anomalies.txt. The person checks in person_checks.txt run on person crops
and need their own pass.

Output is stamped with the encoder identity, same discipline as the .npz
caches: a probe fitted in SigLIP2 space is meaningless against any other
encoder, and loading one silently would produce confident nonsense.

Labels of None (unparseable Moondream answers, about 7% of queries) are
EXCLUDED. An unreadable answer is not evidence either way, and guessing it
would be worse than dropping it.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class _FrameDS(Dataset):
    def __init__(self, paths, pre):
        self.paths, self.pre = paths, pre

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.pre(Image.open(self.paths[i]).convert("RGB"))
        except Exception:                                     # noqa: BLE001
            return self.pre(Image.new("RGB", (224, 224)))


@torch.no_grad()
def embed(paths, model, pre, device, bs=256, workers=12):
    dl = DataLoader(_FrameDS(paths, pre), batch_size=bs, num_workers=workers,
                    pin_memory=True, shuffle=False)
    out, t0, done = [], time.perf_counter(), 0
    for x in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device, enabled=(device == "cuda")):
            e = model.encode_image(x).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        done += x.shape[0]
        if done % (bs * 10) < bs:
            r = done / max(time.perf_counter() - t0, 1e-9)
            print(f"  embed {done}/{len(paths)}  {r:.0f}/s", flush=True)
    return torch.cat(out)


def fit_probe(X, y, epochs=500, lr=0.05, wd=1e-3, device="cpu"):
    """Logistic regression with class balancing.

    Positives are rare. Without pos_weight the fit minimises loss by
    predicting "no" forever - 95% accurate, detects nothing.
    """
    X, y = X.to(device), y.to(device).float()
    lin = nn.Linear(X.shape[1], 1).to(device)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(lin(X).squeeze(-1), y).backward()
        opt.step()
    return lin


def average_precision(scores, labels):
    """The honest metric on imbalanced data - accuracy would look great
    while catching nothing."""
    order = scores.argsort(descending=True)
    y = labels[order].numpy()
    if y.sum() == 0:
        return float("nan")
    cum = np.cumsum(y)
    prec = cum / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default="data/ucf/verified.jsonl")
    ap.add_argument("--out", default="assets/probes")
    ap.add_argument("--min-pos", type=int, default=80,
                    help="skip a check with fewer verified positives")
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()

    from jana2.config import CFG
    from jana2.clip_space import load_clip
    model, pre, _, space_id = load_clip()

    rows = [json.loads(l) for l in open(a.verified) if l.strip()]
    rows = [r for r in rows if Path(r["path"]).exists()]
    print(f"[probes] {len(rows)} verified frames | space {space_id}")

    checks = sorted({k for r in rows for k in r.get("labels", {})})
    print(f"\n{'check':<16}{'pos':>8}{'neg':>8}{'null':>8}")
    for c in checks:
        t = Counter(r["labels"][c] for r in rows if c in r.get("labels", {}))
        print(f"  {c:<14}{t.get(1,0):>8}{t.get(0,0):>8}{t.get(None,0):>8}")

    paths = [r["path"] for r in rows]
    print(f"\n[probes] embedding {len(paths)} frames")
    E = embed(paths, model, pre, CFG.device)

    names, W, B, meta = [], [], [], {}
    print(f"\n{'check':<16}{'n':>7}{'pos':>7}{'AP':>8}"
          f"{'pos mean':>10}{'neg mean':>10}")
    for c in checks:
        idx = [i for i, r in enumerate(rows)
               if r.get("labels", {}).get(c) in (0, 1)]
        if not idx:
            continue
        y = torch.tensor([rows[i]["labels"][c] for i in idx])
        if int(y.sum()) < a.min_pos:
            print(f"  {c:<14} skipped - only {int(y.sum())} positives")
            continue
        X = E[idx]
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(idx), generator=g)
        cut = int(len(idx) * (1 - a.val_frac))
        tr, va = perm[:cut], perm[cut:]

        lin = fit_probe(X[tr], y[tr], device=CFG.device)
        with torch.no_grad():
            s = torch.sigmoid(lin(X[va].to(CFG.device)).squeeze(-1)).cpu()
        yv = y[va]
        apv = average_precision(s, yv)
        mp = s[yv == 1].mean().item() if (yv == 1).any() else float("nan")
        mn = s[yv == 0].mean().item() if (yv == 0).any() else float("nan")

        names.append(c)
        W.append(lin.weight.detach().cpu().numpy()[0])
        B.append(float(lin.bias.detach().cpu()))
        meta[c] = dict(n=len(idx), pos=int(y.sum()), ap=round(apv, 3),
                       mean_pos=round(mp, 3), mean_neg=round(mn, 3))
        print(f"  {c:<14}{len(idx):>7}{int(y.sum()):>7}{apv:>8.3f}"
              f"{mp:>10.3f}{mn:>10.3f}")

    if not names:
        raise SystemExit("no check had enough positives - verify more frames")

    outp = Path(str(a.out) + f"__{space_id.replace('/', '-')}.npz")
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp,
             names=np.array(names, dtype=object),
             W=np.stack(W).astype(np.float32),
             b=np.array(B, dtype=np.float32),
             clip_space=space_id,
             meta=np.array([json.dumps(meta)], dtype=object))
    print(f"\n[probes] {len(names)} probes -> {outp}")
    print("\nRead the AP column. Above ~0.7 the probe is worth using; below "
          "~0.5 it is not, and that check should keep its phrase pair.")
    print("The gap between 'pos mean' and 'neg mean' is the separation the "
          "hand-written phrases never achieved - compare it to the 0.02-0.05 "
          "cosine margins measured this morning.")


if __name__ == "__main__":
    main()
