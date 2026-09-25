"""The contract. Everything downstream reads this and nothing else.

Design rule learned the hard way: keep every stage's opinion separately, and
always keep top-k with scores rather than the argmax. When a caption says the
wrong word you want to be able to point at the exact stage that introduced it.
"""
from dataclasses import dataclass, field, asdict
from typing import Any
import json


def position_bucket(box, W, H) -> str:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / max(W, 1), (y0 + y1) / 2 / max(H, 1)
    col = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    row = "top" if cy < 1 / 3 else ("middle" if cy < 2 / 3 else "bottom")
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


@dataclass
class ObjectSlot:
    idx: int
    box: tuple                       # (x0, y0, x1, y1) absolute pixels
    objectness: float
    source: str                      # "yolo" | "rpn"
    area_frac: float
    position: str
    clip_names: list = field(default_factory=list)   # top-k (name, score)
    slot_names: list = field(default_factory=list)   # reserved: post-memory
    attributes: dict = field(default_factory=dict)   # group -> {label,prob,margin}
    checks: dict = field(default_factory=dict)       # per-object yes/no hits
    importance: float = 0.0
    merged_count: int = 1            # how many raw boxes collapsed into this

    @property
    def name(self):
        return self.clip_names[0][0] if self.clip_names else None

    @property
    def margin(self) -> float:
        """top1 - top2. Low = the model is guessing; use it to abstain."""
        if len(self.clip_names) < 2:
            return 0.0
        return float(self.clip_names[0][1] - self.clip_names[1][1])


@dataclass
class StructuredEvidence:
    image_path: str
    width: int
    height: int
    objects: list = field(default_factory=list)
    scene: dict = field(default_factory=dict)
    relations: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    clip_space: str = ""
    version: str = "0.2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def markdown_table(self) -> str:
        """st.dataframe segfaults on this host (pyarrow). Always render text."""
        rows = ["| # | name | imp | score | margin | objness | src | boxes | pos |",
                "|---|------|-----|-------|--------|---------|-----|-------|-----|"]
        for o in self.objects:
            nm, sc = (o.clip_names[0] if o.clip_names else ("-", 0.0))
            rows.append(
                f"| {o.idx} | {nm} | {o.importance:.2f} | {sc:.3f} | "
                f"{o.margin:.3f} | {o.objectness:.2f} | {o.source} | "
                f"{o.merged_count} | {o.position} |")
        return "\n".join(rows)
