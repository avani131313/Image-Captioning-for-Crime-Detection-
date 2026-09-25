"""Frozen backbone + trained projector. The cheap experiment, run first.

    python -m scripts.distill_projector --data data/distill
    python -m scripts.distill_projector --data data/distill \
        --student MobileCLIP-S1 --epochs 300

THE IDEA

    frozen small backbone  ->  [trained linear map]  ->  SigLIP2's space

SigLIP2's own embeddings are the targets, so there are no human labels
anywhere in this. Every crop you own is training data.

WHY THIS IS MINUTES AND NOT DAYS

Because the backbone never changes, each crop's features are computed
exactly ONCE, for both models, and cached. Training is then a linear map
over cached vectors — no image decoding, no convolutions, no augmentation
in the loop. Hundreds of epochs run in the time full fine-tuning needs for
one. Iterating on the loss weights becomes a coffee break, not a weekend.

WHAT IT CANNOT DO

A linear map REALIGNS information the backbone already encodes. It cannot
invent what is missing. If MobileCLIP does not distinguish "a blade in a
hand" from "a phone in a hand", no projection will. That is exactly what
distill_eval.py measures — and if the margins collapse, the answer is to
unfreeze the backbone (scripts.distill_train), not to train this longer.

The checkpoint format matches distill_train.py, so distill_eval.py reads
either without changes.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from jana2.config import CFG

STUDENTS = {
    "MobileCLIP-S1": ("datacompdr", 21),
    "MobileCLIP-S2": ("datacompdr", 36),
    "TinyCLIP-ViT-40M-32-Text-19M": ("laion400m_e32", 40),
    "MobileCLIP-B": ("datacompdr_lt", 86),
}


@torch.no_grad()
def encode_all(model, pre, root, names, device, bs=64, tag=""):
    """One pass, cached forever. This is the whole speed trick."""
    out, t0 = [], time.perf_counter()
    for i in range(0, len(names), bs):
        ims = []
        for n in names[i:i + bs]:
            try:
                ims.append(pre(Image.open(root / n).convert("RGB")))
            except Exception:                                 # noqa: BLE001
                ims.append(pre(Image.new("RGB", (224, 224))))
        x = torch.stack(ims).to(device)
        e = (model.encode_image(x) if hasattr(model, "encode_image")
             else model(x)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        if i % (bs * 50) == 0:
            print(f"  [{tag}] {i}/{len(names)}")
    print(f"  [{tag}] done in {time.perf_counter()-t0:.0f}s")
    return torch.cat(out)


def load_task_bank(device):
    """Every phrase the pipeline reads — vocab, checks, attributes."""
    banks = []
    for p in (CFG.vocab_path, CFG.anomalies_path,
              CFG.person_checks_path, CFG.attributes_path):
        try:
            z = np.load(p, allow_pickle=True)
            key = "embs" if "embs" in z.files else "vocab"
            banks.append(torch.from_numpy(z[key].astype(np.float32)))
        except Exception as e:                                # noqa: BLE001
            print(f"[task] skipped {Path(p).name} ({e})")
    if not banks:
        return None
    b = torch.cat(banks).to(device)
    print(f"[task] {b.shape[0]} phrases the projector must keep separable")
    return b / b.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill")
    ap.add_argument("--student", default="MobileCLIP-S2", choices=STUDENTS)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0,
                    help="0 = plain linear. >0 inserts one hidden layer, "
                         "which buys a little capacity for a little risk "
                         "of overfitting a small crop set.")
    ap.add_argument("--w-task", type=float, default=2.0)
    ap.add_argument("--task-temp", type=float, default=0.01)
    ap.add_argument("--cache", default=None,
                    help="feature cache path (default <data>/feat_<student>.pt)")
    ap.add_argument("--out", default="checkpoints_distill")
    a = ap.parse_args()

    dev = CFG.device
    root = Path(a.data)
    crops = root / "crops"
    tr = [l.strip() for l in (root / "train.txt").read_text().splitlines() if l.strip()]
    va = [l.strip() for l in (root / "val.txt").read_text().splitlines() if l.strip()]
    print(f"[data] {len(tr)} train | {len(va)} val crops")

    cache = Path(a.cache) if a.cache else root / f"feat_{a.student}.pt"

    # ---- one-time feature extraction --------------------------------- #
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        Ttr, Str, Tva, Sva = d["Ttr"], d["Str"], d["Tva"], d["Sva"]
        s_pre_size = d.get("note", "")
        print(f"[cache] reusing {cache}  {s_pre_size}")
    else:
        import open_clip
        print(f"[teacher] {CFG.clip_model}")
        teacher, _, t_pre = open_clip.create_model_and_transforms(
            CFG.clip_model, pretrained=CFG.clip_pretrained)
        teacher = teacher.to(dev).eval()
        Ttr = encode_all(teacher, t_pre, crops, tr, dev, tag="teacher/train")
        Tva = encode_all(teacher, t_pre, crops, va, dev, tag="teacher/val")
        del teacher
        torch.cuda.empty_cache()

        print(f"[student] {a.student} (frozen)")
        sm, _, s_pre = open_clip.create_model_and_transforms(
            a.student, pretrained=STUDENTS[a.student][0])
        sm = sm.to(dev).eval()
        Str = encode_all(sm, s_pre, crops, tr, dev, tag="student/train")
        Sva = encode_all(sm, s_pre, crops, va, dev, tag="student/val")
        del sm
        torch.cuda.empty_cache()

        torch.save({"Ttr": Ttr, "Str": Str, "Tva": Tva, "Sva": Sva,
                    "note": f"{a.student} -> {CFG.clip_model}"}, cache)
        print(f"[cache] wrote {cache}")

    in_dim, out_dim = Str.shape[1], Ttr.shape[1]
    print(f"[dims] student {in_dim}d -> teacher {out_dim}d")

    # ---- the projector ------------------------------------------------ #
    if a.hidden:
        proj = nn.Sequential(nn.Linear(in_dim, a.hidden), nn.GELU(),
                             nn.Linear(a.hidden, out_dim, bias=False))
    else:
        proj = nn.Linear(in_dim, out_dim, bias=False)
    proj = proj.to(dev)
    n_par = sum(p.numel() for p in proj.parameters())
    print(f"[projector] {n_par/1e6:.2f}M trainable "
          f"(backbone {STUDENTS[a.student][1]}M frozen)")

    task = load_task_bank(dev) if a.w_task > 0 else None
    Ttr, Str = Ttr.to(dev), Str.to(dev)
    Tva_d, Sva_d = Tva.to(dev), Sva.to(dev)

    opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    N = Ttr.shape[0]
    best = -1.0
    Path(a.out).mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        proj.train()
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        for i in range(0, N, a.batch):
            idx = perm[i:i + a.batch]
            s, t = Str[idx], Ttr[idx]
            p = proj(s)
            p = p / p.norm(dim=-1, keepdim=True)

            l_direct = (1.0 - (p * t).sum(-1)).mean()
            l_struct = F.mse_loss(p @ p.T, t @ t.T)
            if task is not None:
                l_task = F.kl_div(((p @ task.T) / a.task_temp).log_softmax(-1),
                                  ((t @ task.T) / a.task_temp).softmax(-1),
                                  reduction="batchmean")
            else:
                l_task = torch.zeros((), device=dev)

            loss = l_direct + l_struct + a.w_task * l_task
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        proj.eval()
        with torch.no_grad():
            pv = proj(Sva_d)
            pv = pv / pv.norm(dim=-1, keepdim=True)
            agree = float((pv * Tva_d).sum(-1).mean())
            raw = float((Sva_d[:, :out_dim] * Tva_d).sum(-1).mean()) \
                if in_dim >= out_dim else float("nan")

        if ep % 20 == 0 or ep == 1:
            print(f"[ep {ep:>4}] loss {tot/N:.4f} | val cos {agree:.4f}")

        if agree > best:
            best = agree
            torch.save({"student": a.student,
                        "pretrained": STUDENTS[a.student][0],
                        "out_dim": out_dim,
                        "in_dim": in_dim,
                        "hidden": a.hidden,
                        "projector_only": True,
                        "teacher": CFG.clip_model,
                        "teacher_pretrained": CFG.clip_pretrained,
                        "val_cos": agree,
                        "epoch": ep,
                        "state": proj.state_dict()},
                       Path(a.out) / "projector_best.pt")

    print(f"\n[done] best val cos {best:.4f}  -> "
          f"{Path(a.out)/'projector_best.pt'}")
    print(f"[size] inference cost {STUDENTS[a.student][1]}M backbone + "
          f"{n_par/1e6:.2f}M projector "
          f"vs 93M SigLIP2")
    print("\nAgreement is NOT the decision. Run distill_eval.py and read the "
          "margin ratios — a high cos with collapsed margins means the "
          "checks stopped working.")


if __name__ == "__main__":
    main()
