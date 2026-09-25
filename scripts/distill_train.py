"""Distil SigLIP2's image tower into a small student, in the SAME space.

    python -m scripts.distill_train --data data/distill
    python -m scripts.distill_train --data data/distill \
        --student MobileCLIP-S2 --epochs 20 --batch 128

WHAT IS AND IS NOT REPLACED

    teacher image tower   93M   <- replaced by the student
    teacher text tower           <- KEPT, untouched, still builds every .npz

Because the text tower is unchanged, the student must land in the teacher's
embedding space. Nothing downstream changes: vocab.txt, anomalies.txt, the
caches, the thresholds all keep working. Swap the encoder, keep the system.

THREE LOSSES, AND WHY THE THIRD ONE MATTERS MOST

  1. direct     cosine(student, teacher) on the same crop.
                Gets the student roughly into the right place.

  2. structure  match the teacher's crop-to-crop similarity matrix within a
                batch. A student can score well on (1) while scrambling
                relative distances — and relative distance is the whole
                signal here.

  3. task       match the teacher's similarity DISTRIBUTION over the actual
                phrases this system reads: vocab.txt, anomalies.txt,
                person_checks.txt, attributes.txt.

                This is the one that protects the pipeline. Every check
                fires on margin = max(pos) - max(neg), and those margins
                live in a 0.02-0.05 band. A student that is "close enough"
                in generic terms can flatten that band to nothing and take
                every anomaly check down with it. Loss 3 optimises the exact
                quantity the system consumes, so it is weighted highest.

Student is initialised from a pretrained small CLIP visual tower, never from
scratch — random init needs millions of images, weight inheritance needs
tens of thousands. A trained linear head maps its width to the teacher's.
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from jana2.config import CFG

# student -> (pretrained tag, approx image-tower params in millions)
STUDENTS = {
    "MobileCLIP-S1": ("datacompdr", 21),
    "MobileCLIP-S2": ("datacompdr", 36),
    "TinyCLIP-ViT-40M-32-Text-19M": ("laion400m_e32", 40),
    "MobileCLIP-B": ("datacompdr_lt", 86),
}


class CropSet(Dataset):
    """Returns the RAW image. Each tower applies its own preprocessing —
    MobileCLIP expects mean=(0,0,0) std=(1,1,1), CLIP expects the usual
    stats, and mixing those up silently wrecks the colours."""

    def __init__(self, root: Path, listing: Path):
        self.root = Path(root) / "crops"
        self.names = [l.strip() for l in
                      Path(listing).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        try:
            return Image.open(self.root / self.names[i]).convert("RGB")
        except Exception:                                     # noqa: BLE001
            return Image.new("RGB", (224, 224), (0, 0, 0))


def collate(batch, t_pre, s_pre):
    return (torch.stack([t_pre(im) for im in batch]),
            torch.stack([s_pre(im) for im in batch]))


class Student(nn.Module):
    """Small visual tower + a linear map into the teacher's dimension."""

    def __init__(self, name, pretrained, out_dim, device):
        super().__init__()
        import open_clip
        m, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.visual = m.visual
        with torch.no_grad():
            probe = self.visual(torch.zeros(1, 3, 224, 224))
            in_dim = probe.shape[-1]
        self.proj = nn.Identity() if in_dim == out_dim else \
            nn.Linear(in_dim, out_dim, bias=False)
        self.to(device)
        print(f"[student] {name} {in_dim}d -> {out_dim}d  "
              f"({sum(p.numel() for p in self.parameters())/1e6:.1f}M)")

    def forward(self, x):
        e = self.proj(self.visual(x)).float()
        return e / e.norm(dim=-1, keepdim=True)


def load_task_text(device):
    """Every text embedding this system actually reads, in one bank."""
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
        raise SystemExit("[task] no caches found — build them first")
    bank = torch.cat(banks).to(device)
    bank = bank / bank.norm(dim=-1, keepdim=True)
    print(f"[task] {bank.shape[0]} phrases the student must keep separable")
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill")
    ap.add_argument("--student", default="MobileCLIP-S2", choices=STUDENTS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w-direct", type=float, default=1.0)
    ap.add_argument("--w-struct", type=float, default=1.0)
    ap.add_argument("--w-task", type=float, default=2.0,
                    help="highest on purpose — see the module docstring")
    ap.add_argument("--task-temp", type=float, default=0.01)
    ap.add_argument("--out", default="checkpoints_distill")
    a = ap.parse_args()

    dev = CFG.device
    import open_clip

    teacher, _, t_pre = open_clip.create_model_and_transforms(
        CFG.clip_model, pretrained=CFG.clip_pretrained)
    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    with torch.no_grad():
        out_dim = teacher.encode_image(torch.zeros(1, 3, 224, 224,
                                                   device=dev)).shape[-1]
    print(f"[teacher] {CFG.clip_model} -> {out_dim}d  "
          f"({sum(p.numel() for p in teacher.visual.parameters())/1e6:.1f}M)")

    student = Student(a.student, STUDENTS[a.student][0], out_dim, dev)
    task_bank = load_task_text(dev) if a.w_task > 0 else None

    root = Path(a.data)
    dl = DataLoader(
        CropSet(root, root / "train.txt"), batch_size=a.batch, shuffle=True,
        num_workers=8, drop_last=True, pin_memory=True,
        collate_fn=lambda b: collate(b, t_pre, student.preprocess))
    vdl = DataLoader(
        CropSet(root, root / "val.txt"), batch_size=a.batch, shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate(b, t_pre, student.preprocess))
    print(f"[data] {len(dl.dataset)} train | {len(vdl.dataset)} val")

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        student.train()
        agg = {"direct": 0.0, "struct": 0.0, "task": 0.0}
        t0 = time.perf_counter()

        for step, (xt, xs) in enumerate(dl):
            xt, xs = xt.to(dev, non_blocking=True), xs.to(dev, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(dev, enabled=dev == "cuda"):
                T = teacher.encode_image(xt).float()
                T = T / T.norm(dim=-1, keepdim=True)

            with torch.amp.autocast(dev, enabled=dev == "cuda"):
                S = student(xs)

                l_direct = (1.0 - (S * T).sum(-1)).mean()

                # relative geometry, not absolute position
                l_struct = F.mse_loss(S @ S.T, T @ T.T)

                # the distribution the pipeline actually reads
                if task_bank is not None:
                    lt = (T @ task_bank.T) / a.task_temp
                    ls = (S @ task_bank.T) / a.task_temp
                    l_task = F.kl_div(ls.log_softmax(-1),
                                      lt.softmax(-1), reduction="batchmean")
                else:
                    l_task = torch.zeros((), device=dev)

                loss = (a.w_direct * l_direct + a.w_struct * l_struct
                        + a.w_task * l_task)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            agg["direct"] += float(l_direct)
            agg["struct"] += float(l_struct)
            agg["task"] += float(l_task)
            if step % 100 == 0:
                print(f"  ep{ep} {step}/{len(dl)}  "
                      f"direct {float(l_direct):.4f}  "
                      f"struct {float(l_struct):.4f}  "
                      f"task {float(l_task):.4f}")

        # ---- validation: plain agreement with the teacher ----------------
        student.eval()
        sims = []
        with torch.no_grad():
            for xt, xs in vdl:
                xt, xs = xt.to(dev), xs.to(dev)
                T = teacher.encode_image(xt).float()
                T = T / T.norm(dim=-1, keepdim=True)
                sims.append((student(xs) * T).sum(-1).cpu())
        agree = float(torch.cat(sims).mean())
        n = len(dl)
        print(f"[ep {ep}] direct {agg['direct']/n:.4f} "
              f"struct {agg['struct']/n:.4f} task {agg['task']/n:.4f} "
              f"| val cos {agree:.4f} | {time.perf_counter()-t0:.0f}s")

        if agree > best:
            best = agree
            torch.save({"student": a.student,
                        "pretrained": STUDENTS[a.student][0],
                        "out_dim": out_dim,
                        "teacher": CFG.clip_model,
                        "teacher_pretrained": CFG.clip_pretrained,
                        "val_cos": agree,
                        "epoch": ep,
                        "state": student.state_dict()},
                       Path(a.out) / "student_best.pt")
            print(f"        saved (val cos {agree:.4f})")

    print(f"\n[done] best teacher agreement {best:.4f}")
    print("Agreement alone does NOT mean the pipeline still works. Run:")
    print("    python -m scripts.distill_eval "
          f"--ckpt {a.out}/student_best.pt --data {a.data}")
    print("and compare the MARGIN distributions before adopting this.")


if __name__ == "__main__":
    main()
