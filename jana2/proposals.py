"""Proposals from several sources, merged by priority.

Sources, in the order they win overlaps:

  yolo   COCO detector. 80 classes, heavily supervised, very accurate on
         them. Labels kept when confident.
  world  YOLO-World, an OPEN-VOCABULARY detector. Give it your own class
         list as text and it detects those directly — auto rickshaw, machete,
         boom barrier. This is the big upgrade over rpn_v2: it returns a box
         AND a name AND a real confidence, instead of a nameless region we
         then have to guess at with CLIP.
  rpn    rpn_v2, the old class-agnostic objectness head. Still useful for
         things absent from every list.
  sam    FastSAM automatic segmentation, masks converted to boxes. Highest
         recall, no names, slowest. Off by default.

Pick sources in config. Each gets its own budget, because their scores are
not on a common scale.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from PIL import Image

from .config import CFG
from .rpn_head import load_rpn

COCO_ALIAS = {
    "cell phone": "mobile phone", "tv": "television", "couch": "sofa",
    "potted plant": "plant", "dining table": "table", "stop sign": "road sign",
    "sports ball": "ball", "wine glass": "glass", "motorbike": "motorcycle",
}


# --------------------------------------------------------------------------- #
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def tv_nms(boxes, scores, thresh):
    if not boxes:
        return []
    from torchvision.ops import nms as _nms
    return _nms(torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32), thresh).tolist()


def suppress_parts(boxes, thresh, frame_area, max_parent_frac=1.0,
                   min_child_ratio=0.55, protected=None):
    """Drop near-duplicate nested boxes; keep small distinct objects.

    A knife inside a person box is a few percent of its area, so it survives.
    A slightly smaller duplicate of the person is ~90%, so it goes.
    """
    protected = protected or set()
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
    keep = []
    for i in order:
        ai = _area(boxes[i])
        if ai <= 0:
            continue
        if i in protected:
            keep.append(i)
            continue
        gone = False
        for j in keep:
            aj = _area(boxes[j])
            if aj / frame_area > max_parent_frac:
                continue
            if ai / max(aj, 1e-9) < min_child_ratio:
                continue
            if _inter(boxes[i], boxes[j]) / ai >= thresh:
                gone = True
                break
        if not gone:
            keep.append(i)
    return keep


# --------------------------------------------------------------------------- #
@dataclass
class ProposalStats:
    raw: int = 0
    by_source_raw: dict = field(default_factory=dict)
    labelled: int = 0
    after_area: int = 0
    after_nms: int = 0
    after_containment: int = 0
    final: int = 0
    by_source_final: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)

    def render(self):
        return (f"  raw            {self.raw:>4}   {self.by_source_raw}\n"
                f"  with a label   {self.labelled:>4}   (detector-named, "
                f"CLIP naming skipped)\n"
                f"  after area     {self.after_area:>4}\n"
                f"  after nms      {self.after_nms:>4}\n"
                f"  after contain  {self.after_containment:>4}\n"
                f"  final          {self.final:>4}   {self.by_source_final}")


class _FeatureNet:
    """Nano YOLO, run only to expose the P3/P4/P5 maps rpn_v2 was trained on."""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.y = YOLO(weights)
        self.net = self.y.model.model
        self.y.model.to(device).eval()
        self.src_idx = list(self.net[-1].f)
        self._buf = {}
        for i in self.src_idx:
            self.net[i].register_forward_hook(self._hook(i))
        with torch.no_grad():
            f = self.feats(torch.zeros(1, 3, CFG.rpn_img_size,
                                       CFG.rpn_img_size, device=device))
        self.in_channels = [x.shape[1] for x in f]
        self.strides = [CFG.rpn_img_size // x.shape[-1] for x in f]
        print(f"[feats] {weights} channels={self.in_channels} "
              f"strides={self.strides}")

    def _hook(self, i):
        def h(_m, _i, out):
            self._buf[i] = out
        return h

    @torch.no_grad()
    def feats(self, t):
        self._buf.clear()
        self.y.model(t)
        return [self._buf[i] for i in self.src_idx]


# --------------------------------------------------------------------------- #
class Proposer:
    def _class_list(self):
        """The open-vocabulary class list, shared by world and yoloe."""
        return [l.strip() for l in
                Path(self.cfg.world_classes_txt).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.det = self.world = self.feat = self.rpn = self.sam = None
        self.yoloe = None
        src = set(cfg.sources)

        if "yolo" in src:
            from ultralytics import YOLO
            self.det = YOLO(cfg.detector_weights)
            self.det.model.to(cfg.device).eval()
            print(f"[yolo] {cfg.detector_weights} @ imgsz={cfg.detector_imgsz}")

        if "world" in src:
            try:
                from ultralytics import YOLOWorld
                self.world = YOLOWorld(cfg.world_weights)
                names = self._class_list()
                self.world.set_classes(names)
                self.world_names = names
                print(f"[world] {cfg.world_weights} with {len(names)} classes")
            except Exception as e:                            # noqa: BLE001
                print(f"[world] unavailable ({e})")
                self.world = None

        if "yoloe" in src:
            try:
                from ultralytics import YOLOE
                self.yoloe = YOLOE(cfg.yoloe_weights)
                names = self._class_list()
                # NOT the same call as YOLO-World. YOLOE wants the text
                # prompt embedding computed and handed in; set_classes(names)
                # alone silently leaves it with no vocabulary.
                self.yoloe.set_classes(names, self.yoloe.get_text_pe(names))
                self.yoloe.model.to(cfg.device).eval()
                print(f"[yoloe] {cfg.yoloe_weights} with {len(names)} classes")
            except Exception as e:                            # noqa: BLE001
                print(f"[yoloe] unavailable ({e})")
                self.yoloe = None

        if "rpn" in src:
            self.feat = _FeatureNet(cfg.feature_weights, cfg.device)
            want = [64, 128, 256]
            if self.feat.in_channels != want:
                print(f"[rpn] channels {self.feat.in_channels} != {want}")
            else:
                try:
                    self.rpn = load_rpn(cfg.rpn_ckpt, want, self.feat.strides,
                                        cfg.device)
                except Exception as e:                        # noqa: BLE001
                    print(f"[rpn] load failed ({e})")

        if "sam" in src:
            try:
                from ultralytics import FastSAM
                self.sam = FastSAM(cfg.sam_weights)
                print(f"[sam] {cfg.sam_weights}")
            except Exception as e:                            # noqa: BLE001
                print(f"[sam] unavailable ({e})")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _from_detector(self, model, image, conf, imgsz, alias=True):
        """PIL in — ultralytics reads a raw ndarray as BGR, PIL as RGB."""
        r = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return [], [], []
        bx = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        names = [r.names[c] for c in cl]
        if alias:
            names = [COCO_ALIAS.get(n, n) for n in names]
        return ([tuple(map(float, b)) for b in bx],
                [float(c) for c in cf], names)

    @torch.no_grad()
    def _from_rpn(self, image):
        if self.rpn is None:
            return [], [], []
        import torchvision.transforms.functional as TF
        W, H = image.size
        S = self.cfg.rpn_img_size
        t = TF.to_tensor(image.resize((S, S))).unsqueeze(0).to(self.cfg.device)
        bx, sc = self.rpn.proposals(self.feat.feats(t), S,
                                    pre_topk=self.cfg.rpn_pre_topk,
                                    iou=self.cfg.rpn_nms,
                                    topk=self.cfg.rpn_budget)
        bx, sc = bx[0].cpu().numpy(), sc[0].cpu().numpy()
        b, s = [], []
        for box, score in zip(bx, sc):
            if score < self.cfg.rpn_min_score:
                continue
            b.append((float(box[0]) * W, float(box[1]) * H,
                      float(box[2]) * W, float(box[3]) * H))
            s.append(float(score))
        return b, s, [None] * len(b)

    @torch.no_grad()
    def _from_sam(self, image):
        if self.sam is None:
            return [], [], []
        r = self.sam.predict(image, imgsz=self.cfg.sam_imgsz,
                             conf=0.25, iou=0.7, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return [], [], []
        bx = r.boxes.xyxy.cpu().numpy()[:self.cfg.sam_budget]
        return ([tuple(map(float, b)) for b in bx],
                [0.5] * len(bx), [None] * len(bx))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, image: Image.Image):
        W, H = image.size
        frame = float(W * H)
        st = ProposalStats()

        got = {}
        if self.det is not None:
            got["yolo"] = self._from_detector(
                self.det, image, self.cfg.yolo_conf, self.cfg.detector_imgsz)
        if self.world is not None:
            got["world"] = self._from_detector(
                self.world, image, self.cfg.world_conf,
                self.cfg.detector_imgsz, alias=False)
        if self.yoloe is not None:
            got["yoloe"] = self._from_detector(
                self.yoloe, image, self.cfg.yoloe_conf,
                self.cfg.detector_imgsz, alias=False)
        if self.rpn is not None:
            got["rpn"] = self._from_rpn(image)
        if self.sam is not None:
            got["sam"] = self._from_sam(image)

        boxes, scores, srcs, labels = [], [], [], []
        for name in self.cfg.sources:                 # priority order
            if name not in got:
                continue
            b, s, l = got[name]
            budget = getattr(self.cfg, f"{name}_budget", 32)
            trust = getattr(self.cfg, f"{name}_trust_conf", 1.1)
            for bb, ss, ll in list(zip(b, s, l))[:budget]:
                boxes.append(bb)
                scores.append(ss)
                srcs.append(name)
                labels.append(ll if (ll and ss >= trust) else None)

        st.raw = len(boxes)
        st.by_source_raw = {k: sum(1 for x in srcs if x == k)
                            for k in dict.fromkeys(srcs)}
        st.labelled = sum(l is not None for l in labels)
        if not boxes:
            return [], [], [], [], st

        keep = [i for i in range(len(boxes))
                if _area(boxes[i]) / frame >= self.cfg.min_area_frac]
        st.after_area = len(keep)

        # earlier sources win overlaps: bump their score for ordering only
        pri = {n: len(self.cfg.sources) - i
               for i, n in enumerate(self.cfg.sources)}
        rank_s = [scores[i] + pri.get(srcs[i], 0) for i in keep]
        keep = [keep[i] for i in tv_nms([boxes[i] for i in keep], rank_s,
                                        self.cfg.nms_iou)]
        st.after_nms = len(keep)

        prot = {n for n, i in enumerate(keep)
                if self.cfg.protect_yolo
                and srcs[i] in ("yolo", "world", "yoloe")}
        keep = [keep[i] for i in suppress_parts(
            [boxes[i] for i in keep], self.cfg.containment_thresh, frame,
            self.cfg.containment_max_parent,
            self.cfg.containment_min_child_ratio, prot)]
        st.after_containment = len(keep)

        keep.sort(key=lambda i: (-pri.get(srcs[i], 0), -scores[i]))
        keep = keep[:self.cfg.max_proposals]

        st.final = len(keep)
        st.by_source_final = {k: sum(1 for i in keep if srcs[i] == k)
                              for k in dict.fromkeys(srcs)}

        return ([boxes[i] for i in keep], [scores[i] for i in keep],
                [srcs[i] for i in keep], [labels[i] for i in keep], st)
