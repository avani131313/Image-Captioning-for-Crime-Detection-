"""Does the student preserve the MARGINS the pipeline runs on?

    python -m scripts.distill_eval --ckpt checkpoints_distill/projector_best.pt

Reads either checkpoint kind — projector-only (frozen backbone + trained
linear map) or a fully fine-tuned student.

WHY AGREEMENT IS THE WRONG NUMBER

"val cos 0.97" is a comfortable figure that can hide the only failure that
matters. This system never reads an embedding. It reads

    margin = max(sim to positive phrases) - max(sim to negative phrases)

and decides. Those margins live in a 0.02-0.05 band. A student at 0.97
agreement can still compress that band toward zero, at which point every
anomaly and person check goes quiet and the app looks perfectly healthy
while detecting nothing at all.

So this compares, on held-out crops, per check:

    teacher margin distribution   vs   student margin distribution

RATIO is the number to read. 1.0 means the check behaves identically.
Below ~0.8 the check has lost discriminating power — either re-tune its
threshold against the new scale, or do not adopt this student.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from jana2.config import CFG
from jana2.anomaly import parse


class StudentModel(nn.Module):
    """Frozen backbone + projector, or a fully fine-tuned student."""

    def __init__(self, ck, device):
        super().__init__()
        import open_clip
        m, _, self.preprocess = open_clip.create_model_and_transforms(
            ck["student"], pretrained=ck["pretrained"])
        self.visual = m.visual

        if ck.get("projector_only"):
            h = ck.get("hidden", 0)
            if h:
                self.proj = nn.Sequential(
                    nn.Linear(ck["in_dim"], h), nn.GELU(),
                    nn.Linear(h, ck["out_dim"], bias=False))
            else:
                self.proj = nn.Linear(ck["in_dim"], ck["out_dim"], bias=False)
            self.proj.load_state_dict(ck["state"])
            kind = "projector-only"
        else:
            with torch.no_grad():
                d = self.visual(torch.zeros(1, 3, 224, 224)).shape[-1]
            self.proj = (nn.Identity() if d == ck["out_dim"]
                         else nn.Linear(d, ck["out_dim"], bias=False))
            self.load_state_dict(ck["state"])
            kind = "full student"

        self.to(device).eval()
        print(f"[student] {ck['student']} ({kind}) ep{ck['epoch']} "
              f"val_cos={ck['val_cos']:.4f}")

    @torch.no_grad()
    def forward(self, x):
        e = self.proj(self.visual(x)).float()
        return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def embed(fn, pre, images, device, bs=64):
    out = []
    for i in range(0, len(images), bs):
        x = torch.stack([pre(im) for im in images[i:i + bs]]).to(device)
        e = fn(x).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


def check_margins(embs, bank, checks):
    """margin per crop per check — exactly as anomaly._pair computes it."""
    out = {}
    for c in checks:
        p = bank[c.pos_start:c.pos_start + len(c.pos)]
        n = bank[c.neg_start:c.neg_start + len(c.neg)]
        sp = (embs @ p.T).max(-1).values
        sn = (embs @ n.T).max(-1).values
        out[c.name] = (sp - sn).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_distill/projector_best.pt")
    ap.add_argument("--data", default="data/distill")
    ap.add_argument("--n", type=int, default=500, help="held-out crops")
    a = ap.parse_args()

    dev = CFG.device
    import open_clip

    teacher, _, t_pre = open_clip.create_model_and_transforms(
        CFG.clip_model, pretrained=CFG.clip_pretrained)
    teacher = teacher.to(dev).eval()

    ck = torch.load(a.ckpt, map_location="cpu")
    student = StudentModel(ck, dev)

    root = Path(a.data)
    names = [l.strip() for l in (root / "val.txt").read_text().splitlines()
             if l.strip()][:a.n]
    imgs = [Image.open(root / "crops" / n).convert("RGB") for n in names]
    print(f"[eval] {len(imgs)} held-out crops\n")

    Te = embed(teacher.encode_image, t_pre, imgs, dev)
    Se = embed(student, student.preprocess, imgs, dev)
    print(f"teacher agreement (cos)   {float((Te*Se).sum(-1).mean()):.4f}\n")

    for path, txt, label in ((CFG.anomalies_path, CFG.anomalies_txt, "FRAME"),
                             (CFG.person_checks_path, CFG.person_checks_txt,
                              "PERSON")):
        try:
            z = np.load(path, allow_pickle=True)
            bank = torch.from_numpy(z["embs"].astype(np.float32))
            checks = parse(txt)
            off = 0
            for c in checks:              # same layout build() wrote
                c.pos_start = off
                off += len(c.pos)
                c.neg_start = off
                off += len(c.neg)
        except Exception as e:                                # noqa: BLE001
            print(f"[{label}] skipped ({e})")
            continue

        mt = check_margins(Te, bank, checks)
        ms = check_margins(Se, bank, checks)

        print(f"{label} CHECKS      teacher      student      ratio")
        print("-" * 58)
        ratios = []
        for c in checks:
            t95 = float(np.percentile(mt[c.name], 95))
            s95 = float(np.percentile(ms[c.name], 95))
            r = s95 / t95 if abs(t95) > 1e-6 else 0.0
            ratios.append(r)
            flag = "  <-- lost" if r < 0.8 else ""
            print(f"  {c.name:<22}{t95:>8.4f}{s95:>13.4f}{r:>11.2f}{flag}")
        if ratios:
            print(f"  {'MEAN':<22}{'':>8}{'':>13}{np.mean(ratios):>11.2f}\n")

    try:
        z = np.load(CFG.vocab_path, allow_pickle=True)
        key = "embs" if "embs" in z.files else "vocab"
        vb = torch.from_numpy(z[key].astype(np.float32))
        vn = [str(x) for x in z["names"]] if "names" in z.files else None
        tt = (Te @ vb.T).argmax(-1)
        ss = (Se @ vb.T).argmax(-1)
        print(f"naming top-1 agreement    {float((tt == ss).float().mean()):.3f}")
        top5 = (Te @ vb.T).topk(5, -1).indices
        in5 = float(sum(ss[i] in top5[i] for i in range(len(ss))) / len(ss))
        print(f"student top-1 in teacher top-5  {in5:.3f}")
        if vn is not None:
            dis = [(vn[int(tt[i])], vn[int(ss[i])])
                   for i in range(len(tt)) if tt[i] != ss[i]][:10]
            if dis:
                print("\nfirst disagreements (teacher -> student):")
                for t, s in dis:
                    print(f"  {t:<28} -> {s}")
    except Exception as e:                                    # noqa: BLE001
        print(f"[vocab] skipped ({e})")

    print("\nAdopt only if the mean margin ratio is near 1.0. High cos with "
          "collapsed margins means the checks stopped working, not that the "
          "student is good.")


if __name__ == "__main__":
    main()
