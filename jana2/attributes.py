"""Second pass: describe an object that has already been named.

Runs only on objects whose name appears in a group's applies_to list, and
scores each group separately. Groups are mutually exclusive, so we softmax
WITHIN a group — "upper_colour" returns one answer, not five.

Nothing is recorded unless the top choice clearly beats the second. For a
crime report, an omitted shirt colour is fine; a guessed one is evidence
that was never there.

Costs almost nothing: the ROI embedding is already computed during naming,
so this is one small matrix multiply per group.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import torch

from .config import CFG
from .clip_space import load_clip

_STRIP = [
    "a photo of a ", "a photo of an ", "a photo of ",
    "a person wearing an ", "a person wearing a ", "a person wearing ",
    "a person carrying an ", "a person carrying a ", "a person carrying ",
    "a person holding an ", "a person holding a ", "a person holding ",
    "a person with an ", "a person with a ", "a person with ",
    "a person ", "a rider ", "a motorcycle with ", "a vehicle with ",
    "an ", "a ",
]


def short_label(phrase: str) -> str:
    p = phrase.lower().strip()
    for s in _STRIP:
        if p.startswith(s):
            return p[len(s):].strip()
    return p


@dataclass
class Group:
    name: str
    applies_to: set
    phrases: list
    labels: list
    start: int = 0          # row offset into the shared embedding matrix

    @property
    def n(self):
        return len(self.phrases)


# --------------------------------------------------------------------------- #
def parse(path: Path) -> list[Group]:
    """Read attributes.txt -> [Group]. Header: [name] applies_to=a,b,c"""
    groups, cur = [], None
    header = re.compile(r"^\[(?P<name>[^\]]+)\]\s*(applies_to=(?P<to>.*))?$")
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = header.match(line)
        if m:
            to = m.group("to") or ""
            cur = Group(name=m.group("name").strip(),
                        applies_to={t.strip().lower()
                                    for t in to.split(",") if t.strip()},
                        phrases=[], labels=[])
            groups.append(cur)
            continue
        if cur is not None:
            cur.phrases.append(line)
            cur.labels.append(short_label(line))
    return [g for g in groups if g.n >= 2]      # a 1-option group is useless


@torch.no_grad()
def build(txt_path: Path = None, out_path: Path = None) -> None:
    """Encode every attribute phrase once and cache it."""
    txt_path = Path(txt_path or CFG.attributes_txt)
    out_path = Path(out_path or CFG.attributes_path)
    model, _, tok, space_id = load_clip()

    groups = parse(txt_path)
    all_phrases, meta = [], []
    off = 0
    for g in groups:
        g.start = off
        all_phrases += g.phrases
        off += g.n
        meta.append((g.name, ",".join(sorted(g.applies_to)), g.start, g.n))

    embs = []
    for i in range(0, len(all_phrases), 256):
        t = tok(all_phrases[i:i + 256]).to(CFG.device)
        e = model.encode_text(t).float()
        embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    embs = np.concatenate(embs).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             embs=embs,
             labels=np.array([l for g in groups for l in g.labels], dtype=object),
             meta=np.array(meta, dtype=object),
             clip_space=space_id)
    print(f"[attr] {len(groups)} groups, {len(all_phrases)} phrases "
          f"-> {out_path} (space {space_id})")


# --------------------------------------------------------------------------- #
class AttributeBank:
    def __init__(self, path: Path = None, cfg=CFG):
        self.cfg = cfg
        path = Path(path or cfg.attributes_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python -m scripts.build_attributes")
        z = np.load(path, allow_pickle=True)

        model, _, _, space_id = load_clip()
        if str(z["clip_space"]) != space_id:
            raise RuntimeError(
                f"attribute/encoder mismatch: file built in '{z['clip_space']}' "
                f"but active encoder is '{space_id}'. Rebuild.")

        self.embs = torch.from_numpy(z["embs"].astype(np.float32)).to(cfg.device)
        labels = list(z["labels"])
        self.groups = []
        for name, to, start, n in z["meta"]:
            start, n = int(start), int(n)
            self.groups.append(Group(
                name=str(name),
                applies_to={t for t in str(to).split(",") if t},
                phrases=[], labels=labels[start:start + n], start=start))
            self.groups[-1].phrases = [None] * n

        ls = getattr(model, "logit_scale", None)
        self.scale = ls.detach().exp() if ls is not None else None
        print(f"[attr] {len(self.groups)} groups loaded "
              f"({sum(g.n for g in self.groups)} phrases)")

    def groups_for(self, obj_name: str):
        n = (obj_name or "").lower()
        return [g for g in self.groups if n in g.applies_to]

    @torch.no_grad()
    def describe(self, obj_name: str, roi_emb: torch.Tensor) -> dict:
        """roi_emb: [D] unit-norm. Returns {group: {label, prob, margin}}."""
        out = {}
        for g in self.groups_for(obj_name):
            sub = self.embs[g.start:g.start + g.n]
            sims = roi_emb @ sub.T
            logits = sims * self.scale if self.scale is not None else sims
            probs = logits.softmax(-1)               # exclusive within group
            top = torch.topk(probs, min(2, g.n))
            p1 = float(top.values[0])
            p2 = float(top.values[1]) if g.n > 1 else 0.0
            if (p1 - p2) < self.cfg.attr_min_margin:
                continue                              # too close to call — omit
            out[g.name] = {"label": g.labels[int(top.indices[0])],
                           "prob": round(p1, 4),
                           "margin": round(p1 - p2, 4)}
        return out
