"""Naming, attributes, per-object checks, merge, rank, relations, anomalies.

Order inside build_evidence():

    crops -> CLIP embeddings        (once; every later pass reuses them)
      -> names          softmax over the vocabulary
      -> attributes     softmax within each exclusive group
      -> person checks  contrastive pairs on person crops (catches a weapon
                        without ever detecting the weapon)
      -> merge          geometry-first duplicate collapse
      -> anomalies      gated frame checks, given object context
      -> rank           explicit importance policy
      -> relations      geometry, derived AFTER ranking (rules use o.idx)

Naming policy: a confident detector label beats CLIP zero-shot on a small
crop, so a trusted YOLO/World label is kept and CLIP only names the rest.
"""
from __future__ import annotations
import time
import torch
from PIL import Image

from .config import CFG
from .clip_space import load_clip, load_vocab
from .evidence import ObjectSlot, StructuredEvidence, position_bucket
from .postprocess import merge_duplicates, rank, category, canon


def _crop(image, box, pad):
    W, H = image.size
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - pw), max(0, y0 - ph),
                       min(W, x1 + pw), min(H, y1 + ph)))


class Namer:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.model, self.preprocess, _, self.space = load_clip()
        self.names, self.vocab = load_vocab(cfg.vocab_path)
        self.vocab_t = torch.from_numpy(self.vocab).to(cfg.device)

        ls = getattr(self.model, "logit_scale", None)
        lb = getattr(self.model, "logit_bias", None)
        self.logit_scale = ls.detach() if ls is not None else None
        self.logit_bias = lb.detach() if lb is not None else None
        print(f"[namer] {len(self.names)} names | readout={cfg.name_readout}")

        # scene, attrs, person checks, anomaly checks are mandatory — if any
        # cache is missing or was built with a different CLIP encoder, this
        # raises immediately with what's wrong instead of quietly running
        # with that subsystem off. Rebuild missing ones with:
        #   python -m scripts.build_scenes
        #   python -m scripts.build_attributes
        #   python -m scripts.build_anomalies
        #   python -m scripts.build_person_checks
        have = cfg.caches_present()
        missing = [k for k in ("scenes", "attributes", "anomalies",
                                "person_checks") if not have[k]]
        if missing:
            raise RuntimeError(
                f"[namer] required caches missing for clip_model="
                f"{cfg.clip_model!r}: {missing}. Build them first (see "
                f"scripts.build_<name> for each), then restart.")

        sn, se = load_vocab(cfg.scenes_path)
        self.scene_names = sn
        self.scene_t = torch.from_numpy(se).to(cfg.device)
        print(f"[namer] {len(sn)} scene labels")

        self.attrs = _attr_bank(cfg)
        self.person_checks = _bank(cfg.person_checks_path, cfg, "person")
        self.frame_checks = _bank(cfg.anomalies_path, cfg, "anomaly")
        print("[namer] attrs=ON  person_checks=ON  anomaly=ON  scene=ON")

    @torch.no_grad()
    def embed_rois(self, image, boxes):
        if not boxes:
            return torch.empty(0, self.vocab_t.shape[1], device=self.cfg.device)
        crops = [self.preprocess(_crop(image, b, self.cfg.roi_pad)) for b in boxes]
        e = self.model.encode_image(torch.stack(crops).to(self.cfg.device)).float()
        return e / e.norm(dim=-1, keepdim=True)

    def _readout(self, sims):
        if self.logit_scale is None:
            return sims
        logits = sims * self.logit_scale.exp()
        if self.cfg.name_readout == "sigmoid" and self.logit_bias is not None:
            return torch.sigmoid(logits + self.logit_bias)
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def topk(self, embs, bank=None, k=None):
        bank = self.vocab_t if bank is None else bank
        names = self.names if bank is self.vocab_t else self.scene_names
        k = k or self.cfg.topk_names
        if embs.numel() == 0:
            return []
        probs = self._readout(embs @ bank.T)
        vals, idxs = probs.topk(min(k, probs.shape[1]), dim=-1)
        return [[(names[int(j)], float(v)) for v, j in zip(vr, ir)]
                for vr, ir in zip(vals.cpu(), idxs.cpu())]

    @torch.no_grad()
    def scene(self, image):
        """Returns (topk list, best RAW cosine).

        The raw cosine is the only thing that can say "nothing in this list
        matches". Softmax always elects a winner, and through SigLIP's
        logit_scale (~100) that winner looks confident whether or not the
        real scene is in the vocabulary at all — which is how a parked
        motorcycle became "a road accident scene".
        """
        if self.scene_t is None:
            return [], 0.0
        e = self.model.encode_image(
            self.preprocess(image).unsqueeze(0).to(self.cfg.device)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        best = float((e @ self.scene_t.T).max())
        return self.topk(e, bank=self.scene_t), best


def _attr_bank(cfg):
    from .attributes import AttributeBank
    return AttributeBank(cfg=cfg)


def _bank(path, cfg, label):
    from .anomaly import CheckBank
    return CheckBank(path, cfg=cfg, label=label)


# --------------------------------------------------------------------------- #
def build_evidence(image_path, image, boxes, scores, srcs, labels, namer):
    W, H = image.size
    t = {}

    t0 = time.perf_counter()
    embs = namer.embed_rois(image, boxes)
    t["clip_roi"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    clip_names = namer.topk(embs)
    t["naming"] = (time.perf_counter() - t0) * 1000

    objs = []
    for i, (b, s, src) in enumerate(zip(boxes, scores, srcs)):
        lab = labels[i] if labels else None
        if lab:                                  # trusted detector label wins
            alt = [x for x in clip_names[i] if x[0] != lab][:3]
            names = [(lab, round(float(s), 4))] + [(n, round(sc, 4))
                                                   for n, sc in alt]
            named_by = "detector"
        else:
            names = [(n, round(sc, 4)) for n, sc in clip_names[i]]
            named_by = "clip"

        objs.append(ObjectSlot(
            idx=i + 1,
            box=tuple(round(float(v), 1) for v in b),
            objectness=round(float(s), 4),
            source=src,
            area_frac=round(((b[2] - b[0]) * (b[3] - b[1])) / (W * H), 4),
            position=position_bucket(b, W, H),
            clip_names=names,
            named_by=named_by,
        ))

    # ---- attributes --------------------------------------------------
    t0 = time.perf_counter()
    if namer.attrs is not None:
        for i, o in enumerate(objs):
            if o.name:
                o.attributes = namer.attrs.describe(o.name, embs[i])
    t["attributes"] = (time.perf_counter() - t0) * 1000

    # ---- per-person checks -------------------------------------------
    t0 = time.perf_counter()
    if namer.person_checks is not None and objs:
        pidx = [i for i, o in enumerate(objs) if category(o.name) == "person"]
        if pidx:
            for slot, h in zip(pidx, namer.person_checks.scan_embs(embs[pidx])):
                objs[slot].checks = h
    t["person_checks"] = (time.perf_counter() - t0) * 1000

    # ---- merge duplicates --------------------------------------------
    t0 = time.perf_counter()
    objs = merge_duplicates(objs, CFG.dedup_iou, CFG.dedup_contain,
                            CFG.dedup_same_object_iou)
    t["merge"] = (time.perf_counter() - t0) * 1000

    # ---- frame anomalies, gated on what was actually found -----------
    t0 = time.perf_counter()
    ctx = {
        # exact name, its synonym-canonical form, AND its category — so
        # requires=vehicle catches "car", "damaged vehicle", "damaged car"
        # or anything else that will ever be categorised as a vehicle,
        # instead of anomalies.txt needing to enumerate every phrasing.
        "names": ({(o.name or "").lower() for o in objs}
                  | {canon(o.name) for o in objs}
                  | {category(o.name) for o in objs}),
        "n_people": sum(1 for o in objs if category(o.name) == "person"),
    }
    anomalies = (namer.frame_checks.scan_frame(image, ctx)
                 if namer.frame_checks is not None else [])
    t["anomaly"] = (time.perf_counter() - t0) * 1000

    hot = [a["box"] for a in anomalies
           if a.get("triggered") and a.get("where") not in ("full", "gated")]
    objs = rank(objs, W, H, hot)

    # ---- relations, AFTER ranking (rules reference o.idx) ------------
    t0 = time.perf_counter()
    rels = []
    if CFG.use_relations:
        from .relations import derive
        rels = derive(objs, W, H, CFG.max_relations)
    t["relations"] = (time.perf_counter() - t0) * 1000

    sc, sc_cos = namer.scene(image)
    return StructuredEvidence(
        image_path=str(image_path), width=W, height=H, objects=objs,
        scene={"clip_names": [(n, round(s, 4)) for n, s in sc[0]],
               "cos": round(sc_cos, 4)} if sc else {},
        relations=rels,
        anomalies=anomalies,
        timings_ms={k: round(v, 1) for k, v in t.items()},
        clip_space=namer.space,
    )
