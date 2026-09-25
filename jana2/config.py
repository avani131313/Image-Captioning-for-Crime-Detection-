"""JANA2 config.

Text-embedding caches are named per CLIP model — `vocab__MobileCLIP-S2.npz`,
`vocab__ViT-B-16-SigLIP2.npz` and so on. That way you can switch encoders in
the UI without the caches fighting each other, and comparing two models is a
dropdown rather than a rebuild.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# The only CLIP encoders offered. Image-tower params are what run per image;
# the text tower is used once by the build scripts and never again.
CLIP_CHOICES = {
    "MobileCLIP-S1":    ("datacompdr",    21),
    "MobileCLIP-S2":    ("datacompdr",    36),
    "TinyCLIP-ViT-40M-32-Text-19M": ("laion400m_e32", 40),
    "MobileCLIP-B":     ("datacompdr_lt", 86),
    "ViT-B-16-SigLIP2": ("webli",         93),
}


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- CLIP -------------------------------------------------------------
    # Sticking with SigLIP2 for now — biggest image tower of the candidates,
    # but it's the one already tuned/calibrated (softmax temp, margins) and
    # proven in testing. Revisit TinyCLIP later if the param budget bites.
    clip_model: str = "ViT-B-16-SigLIP2"
    clip_pretrained: str = "webli"

    # ---- detector ----------------------------------------------------------
    # yolo26n (trusted COCO labels, nano) + YOLO-World small (open vocab).
    # rpn_v2 dropped — it was the source of most bad/nameless guesses.
    sources: tuple = ("yolo", "world")
    detector_weights: str = "yolo26n.pt"
    yolo_conf: float = 0.10
    yolo_trust_conf: float = 0.40
    yolo_budget: int = 32

    world_weights: str = "yolov8s-worldv2.pt"
    world_classes_txt: Path = ASSETS / "world_classes.txt"
    detector_imgsz: int = 640
    world_conf: float = 0.08
    world_trust_conf: float = 0.15     # above this, keep the detector's label
    world_budget: int = 48

    # ---- YOLOE (open vocabulary, newer than YOLO-World) --------------------
    # Shares world_classes.txt — same class list, different detector, so the
    # two are directly comparable. Note the API differs: YOLOE needs the text
    # embedding passed in explicitly via get_text_pe(), YOLO-World does not.
    yoloe_weights: str = "yoloe-26n-seg.pt"
    yoloe_conf: float = 0.08
    yoloe_trust_conf: float = 0.15
    yoloe_budget: int = 48

    # ---- kept for compatibility; rpn no longer wired in ---------------------
    use_yolo: bool = True
    use_rpn: bool = False
    feature_weights: str = "yolo26n.pt"
    rpn_ckpt: Path = ROOT / "checkpoints_rpn" / "rpn_ep2.pt"
    rpn_img_size: int = 640
    rpn_pre_topk: int = 300
    rpn_nms: float = 0.6
    rpn_min_score: float = 0.20
    rpn_budget: int = 32
    sam_weights: str = "FastSAM-s.pt"
    sam_imgsz: int = 1024
    sam_budget: int = 40

    # ---- geometric cleanup -------------------------------------------------
    max_proposals: int = 48
    min_area_frac: float = 0.0015
    nms_iou: float = 0.6
    containment_thresh: float = 0.90
    containment_min_child_ratio: float = 0.55   # small held objects survive
    containment_max_parent: float = 0.60
    protect_yolo: bool = True

    # ---- naming -----------------------------------------------------------
    name_readout: str = "softmax"
    roi_pad: float = 0.08
    topk_names: int = 5

    # ---- attributes -------------------------------------------------------
    use_attributes: bool = True
    attr_min_margin: float = 0.10

    # ---- yes/no checks ----------------------------------------------------
    anomaly_grids: tuple = (1, 2)      # full + 2x2 = 5 crops. (1,2,3) = 14.
    # Paired with the zero-based rescale in anomaly._pair(). Temperature and
    # baseline have to move together: rescaling to start at 0 while leaving
    # T=20 meant threshold 0.60 demanded a cosine margin of 0.069, and
    # nothing in this embedding space gets near that — every check went
    # silent. At T=40 the observed noise floor (~0.022) scores 0.41 and is
    # rejected, while 0.035+ fires.
    check_temperature: float = 40.0
    # Independent of the per-check threshold on purpose.
    check_min_cos_margin: float = 0.03

    # ---- scene ------------------------------------------------------------
    # A softmax over 40 labels ALWAYS elects a winner, so the margin test
    # alone cannot tell "this is a mall entrance" from "none of these 40
    # places is what I am looking at". The raw cosine can. Below the floor
    # the caption simply says nothing about the location, which is correct:
    # an unknown place is not a licence to invent one.
    scene_min_cos: float = 0.10
    scene_min_margin: float = 0.05

    # ---- relations --------------------------------------------------------
    use_relations: bool = True
    max_relations: int = 16

    # ---- merging + caption -------------------------------------------------
    dedup_same_object_iou: float = 0.55
    dedup_iou: float = 0.30
    dedup_contain: float = 0.70
    caption_min_score: float = 0.03
    caption_min_objectness: float = 0.12

    # ---- source text files (shared by every encoder) ----------------------
    vocab_txt: Path = ASSETS / "vocab.txt"
    scenes_txt: Path = ASSETS / "scenes.txt"
    attributes_txt: Path = ASSETS / "attributes.txt"
    anomalies_txt: Path = ASSETS / "anomalies.txt"
    person_checks_txt: Path = ASSETS / "person_checks.txt"

    data_dir: Path = ROOT / "data"
    out_dir: Path = ROOT / "outputs"

    # ---- per-encoder cache paths, filled in below -------------------------
    def __post_init__(self):
        self._refresh_cache_paths()

    def _refresh_cache_paths(self):
        tag = self.clip_model.replace("/", "-").replace(" ", "")
        self.vocab_path = ASSETS / f"vocab__{tag}.npz"
        self.scenes_path = ASSETS / f"scenes__{tag}.npz"
        self.attributes_path = ASSETS / f"attributes__{tag}.npz"
        self.anomalies_path = ASSETS / f"anomalies__{tag}.npz"
        self.person_checks_path = ASSETS / f"person_checks__{tag}.npz"

    def set_clip(self, model: str, pretrained: str | None = None):
        """Switch encoder and point the caches at that encoder's files."""
        self.clip_model = model
        self.clip_pretrained = pretrained or CLIP_CHOICES.get(
            model, (self.clip_pretrained, 0))[0]
        self._refresh_cache_paths()

    def caches_present(self) -> dict:
        return {
            "vocab": self.vocab_path.exists(),
            "scenes": self.scenes_path.exists(),
            "attributes": self.attributes_path.exists(),
            "anomalies": self.anomalies_path.exists(),
            "person_checks": self.person_checks_path.exists(),
        }

    def missing_caches(self) -> list:
        return [k for k, v in self.caches_present().items() if not v]


CFG = Config()
