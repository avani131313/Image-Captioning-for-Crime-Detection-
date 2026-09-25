"""Yes/no checks — with the three things that stopped them being useless.

1. TEMPERATURE. A contrastive pair scored through CLIP's own logit_scale
   (~100) turns a cosine gap of 0.05 into 99% confidence. That is calibrated
   for retrieval across a large batch, not for a 2-way comparison. We use a
   small fixed temperature instead, so the number reflects the actual gap.

2. AN ABSOLUTE FLOOR. Preferring "no helmet" over "wearing a helmet" is not
   evidence of a motorcycle. A check must ALSO clear a raw cosine margin, so
   two equally-bad matches cannot produce an alert.

3. GATING. `requires=motorcycle` means the check does not run unless a
   motorcycle was detected. `requires_people=6` needs six people.
   `tiles=no` restricts a check to the whole frame — darkness in one tile of
   a bright image is a dark jacket, not a dark scene.

Header syntax:
    [name] severity=high threshold=0.70 requires=motorcycle,scooter tiles=no
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import torch
from PIL import Image

from .config import CFG
from .clip_space import load_clip


@dataclass
class Check:
    name: str
    severity: str = "medium"
    threshold: float = 0.7
    requires: tuple = ()          # object names that must be present
    requires_people: int = 0      # minimum person count
    tiles: bool = True            # False = whole frame only
    pos: list = field(default_factory=list)
    neg: list = field(default_factory=list)
    pos_start: int = 0
    neg_start: int = 0


_HDR = re.compile(r"^\[(?P<name>[^\]]+)\]\s*(?P<kv>.*)$")


def parse(path: Path) -> list[Check]:
    checks, cur = [], None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _HDR.match(line)
        if m:
            kv = dict(re.findall(r"(\w+)=([^\s]+)", m.group("kv") or ""))
            cur = Check(
                name=m.group("name").strip(),
                severity=kv.get("severity", "medium"),
                threshold=float(kv.get("threshold", 0.7)),
                requires=tuple(x.strip().lower()
                               for x in kv.get("requires", "").split(",")
                               if x.strip()),
                requires_people=int(kv.get("requires_people", 0)),
                tiles=kv.get("tiles", "yes").lower() not in ("no", "false", "0"),
            )
            checks.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("+"):
            cur.pos.append(line[1:].strip())
        elif line.startswith("-"):
            cur.neg.append(line[1:].strip())
    return [c for c in checks if c.pos and c.neg]


@torch.no_grad()
def build(txt_path: Path, out_path: Path) -> None:
    model, _, tok, space_id = load_clip()
    checks = parse(txt_path)

    phrases, meta = [], []
    for c in checks:
        c.pos_start = len(phrases)
        phrases += c.pos
        c.neg_start = len(phrases)
        phrases += c.neg
        meta.append((c.name, c.severity, c.threshold,
                     ",".join(c.requires), c.requires_people,
                     "1" if c.tiles else "0",
                     c.pos_start, len(c.pos), c.neg_start, len(c.neg)))

    embs = []
    for i in range(0, len(phrases), 256):
        t = tok(phrases[i:i + 256]).to(CFG.device)
        e = model.encode_text(t).float()
        embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    embs = np.concatenate(embs).astype(np.float32)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embs=embs, meta=np.array(meta, dtype=object),
             clip_space=space_id)
    print(f"[checks] {len(checks)} checks, {len(phrases)} phrases -> {out_path}")


# --------------------------------------------------------------------------- #
def tiles(image: Image.Image, grids=(1, 2, 3)):
    W, H = image.size
    out = []
    for g in grids:
        if g == 1:
            out.append((image, "full", (0, 0, W, H)))
            continue
        tw, th = W / g, H / g
        for r in range(g):
            for c in range(g):
                box = (int(c * tw), int(r * th), int((c + 1) * tw), int((r + 1) * th))
                out.append((image.crop(box), f"{g}x{g}:r{r}c{c}", box))
    return out


class CheckBank:
    def __init__(self, path: Path, cfg=CFG, label="checks"):
        path = Path(path)
        self.cfg, self.label = cfg, label
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — build it first")
        z = np.load(path, allow_pickle=True)

        self.model, self.preprocess, _, space_id = load_clip()
        if str(z["clip_space"]) != space_id:
            raise RuntimeError(f"{label}: built in '{z['clip_space']}', "
                               f"active encoder '{space_id}'. Rebuild.")

        self.embs = torch.from_numpy(z["embs"].astype(np.float32)).to(cfg.device)
        self.checks = []
        for row in z["meta"]:
            name, sev, thr, req, reqp, tl, ps, pn, ns, nn_ = row
            c = Check(name=str(name), severity=str(sev), threshold=float(thr),
                      requires=tuple(x for x in str(req).split(",") if x),
                      requires_people=int(reqp), tiles=str(tl) == "1")
            c.pos_start, c.neg_start = int(ps), int(ns)
            c.pos, c.neg = [None] * int(pn), [None] * int(nn_)
            self.checks.append(c)
        print(f"[{label}] {len(self.checks)} checks loaded")

    # ---------------------------------------------------------------- #
    def _gated(self, c: Check, ctx: dict) -> bool:
        """False = do not even run this check on this image."""
        if not ctx:
            return True
        if c.requires:
            present = ctx.get("names", set())
            if not any(r in present for r in c.requires):
                return False
        if c.requires_people and ctx.get("n_people", 0) < c.requires_people:
            return False
        return True

    @torch.no_grad()
    def _pair(self, embs: torch.Tensor, c: Check):
        """Returns (prob, cos_margin) per row. Temperature, not logit_scale.

        The scale has to start at zero. Plain sigmoid(margin * T) returns
        0.5 when the positives and negatives match a crop EQUALLY WELL —
        i.e. when there is no evidence at all. A threshold of 0.60 against
        that baseline fires on a cosine margin of 0.022, which is noise,
        and is why every person came back "weapon held 61%".

        Rescaling so no-evidence reads as 0.0 makes a threshold mean what
        everyone reading anomalies.txt assumed it meant:

        At check_temperature=40 (see config.py — the two MUST be tuned
        together):

            margin 0.022 ->  0.41     the observed noise floor, rejected
            margin 0.030 ->  0.54     still rejected at threshold 0.60
            margin 0.035 ->  0.60     fires
            margin 0.050 ->  0.76     fires clearly

        These numbers are calibrated against noise, not against known true
        positives — run scripts/caliberate.py once you have a folder of
        event images to set them properly.
        """
        p = self.embs[c.pos_start:c.pos_start + len(c.pos)]
        n = self.embs[c.neg_start:c.neg_start + len(c.neg)]
        sp = (embs @ p.T).max(-1).values      # best positive, not the mean
        sn = (embs @ n.T).max(-1).values      # best negative
        margin = sp - sn
        raw = torch.sigmoid(margin * self.cfg.check_temperature)
        prob = (2.0 * (raw - 0.5)).clamp(min=0.0)
        return prob, margin

    @torch.no_grad()
    def scan_frame(self, image: Image.Image, context: dict | None = None,
                   grids=None) -> list[dict]:
        grids = tuple(grids or self.cfg.anomaly_grids)
        tl = tiles(image, grids)
        batch = torch.stack([self.preprocess(c) for c, _, _ in tl]).to(self.cfg.device)
        e = self.model.encode_image(batch).float()
        e = e / e.norm(dim=-1, keepdim=True)

        results = []
        for c in self.checks:
            if not self._gated(c, context or {}):
                results.append({
                    "name": c.name, "severity": c.severity, "prob": 0.0,
                    "cos_margin": 0.0, "threshold": c.threshold,
                    "where": "gated", "box": (0, 0, 0, 0), "full_frame": 0.0,
                    "triggered": False, "skipped": "requirement not met"})
                continue

            prob, margin = self._pair(e, c)
            if not c.tiles:                       # whole frame only
                prob, margin = prob[:1], margin[:1]
            k = int(prob.argmax())
            fired = (float(prob[k]) >= c.threshold
                     and float(margin[k]) >= self.cfg.check_min_cos_margin)
            results.append({
                "name": c.name, "severity": c.severity,
                "prob": round(float(prob[k]), 4),
                "cos_margin": round(float(margin[k]), 4),
                "threshold": c.threshold,
                "where": tl[k][1], "box": tl[k][2],
                "full_frame": round(float(prob[0]), 4),
                "triggered": bool(fired), "skipped": ""})
        results.sort(key=lambda r: (not r["triggered"], -r["prob"]))
        return results

    @torch.no_grad()
    def scan_embs(self, embs: torch.Tensor) -> list[dict]:
        """Score already-computed crop embeddings. One dict of hits per row."""
        if embs.numel() == 0:
            return []
        out = [dict() for _ in range(embs.shape[0])]
        for c in self.checks:
            prob, margin = self._pair(embs, c)
            for i in range(embs.shape[0]):
                if (float(prob[i]) >= c.threshold
                        and float(margin[i]) >= self.cfg.check_min_cos_margin):
                    out[i][c.name] = {"prob": round(float(prob[i]), 4),
                                      "cos_margin": round(float(margin[i]), 4),
                                      "severity": c.severity}
        return out

    @torch.no_grad()
    def scan_embs_full(self, embs: torch.Tensor) -> list[dict]:
        if embs.numel() == 0:
            return []
        rows = [dict() for _ in range(embs.shape[0])]
        for c in self.checks:
            prob, _ = self._pair(embs, c)
            for i in range(embs.shape[0]):
                rows[i][c.name] = round(float(prob[i]), 4)
        return rows


AnomalyBank = CheckBank
