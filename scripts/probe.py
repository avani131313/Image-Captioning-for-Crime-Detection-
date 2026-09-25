"""End-to-end perception probe: image -> proposals -> names -> attributes.

    python -m scripts.probe data/eval/factory.jpg --overlay
    python -m scripts.probe gate.jpg --overlay --json outputs/gate.json
    python -m scripts.probe road.jpg --no-rpn --yolo yolo26n.pt --conf 0.05

Diagnostic CLI, not product code. It answers one question: does the
perception stack see the right things, before any language model exists to
blame for getting them wrong.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from PIL import Image, ImageDraw

from jana2.config import CFG

_PALETTE = {"yolo": (255, 90, 60), "rpn": (60, 170, 255)}


def draw_overlay(image, evidence, out_path: Path, max_boxes=30):
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    for o in evidence.objects[:max_boxes]:
        col = _PALETTE.get(o.source, (200, 200, 200))
        x0, y0, x1, y1 = o.box
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        label = f"{o.idx}:{o.name or '?'}"
        tb = d.textbbox((x0, y0), label)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=col)
        d.text((x0, y0), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def run(paths, dump_json=None, overlay=False, topk_show=3):
    from jana2.proposals import Proposer
    from jana2.naming import Namer, build_evidence

    proposer, namer = Proposer(), Namer()

    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"!! {p} not found")
            continue
        img = Image.open(p).convert("RGB")

        t0 = time.perf_counter()
        boxes, scores, srcs, stats = proposer(img)
        t_prop = (time.perf_counter() - t0) * 1000

        ev = build_evidence(p, img, boxes, scores, srcs, namer)
        ev.timings_ms["proposals"] = round(t_prop, 1)

        print(f"\n{'='*76}\n{p}   {ev.width}x{ev.height}   space={ev.clip_space}")
        print(f"[yolo={CFG.yolo_weights if CFG.use_yolo else 'off'} "
              f"conf={CFG.yolo_conf} | rpn={'on' if CFG.use_rpn else 'off'} "
              f"topk={CFG.rpn_pre_topk} | contain={CFG.containment_thresh} "
              f"maxparent={CFG.containment_max_parent}]")

        print("\nproposal funnel:")
        print(stats.render())

        if ev.scene.get("clip_names"):
            print("\nscene:", ", ".join(
                f"{n} {s:.3f}" for n, s in ev.scene["clip_names"][:4]))
        print()
        print(ev.markdown_table())

        print("\nper object — names, then attributes:")
        for o in ev.objects[:20]:
            alt = "  ".join(f"{n}:{s:.3f}" for n, s in o.clip_names[:topk_show])
            flag = "   <-- LOW MARGIN" if o.margin < 0.05 else ""
            print(f"  #{o.idx:<3}[{o.source:<4}] {alt}{flag}")
            if o.attributes:
                for g, v in o.attributes.items():
                    print(f"        {g:<18} {v['label']:<32} "
                          f"p={v['prob']:.2f} m={v['margin']:.2f}")

        print("\ntimings (ms):", ev.timings_ms)

        if overlay:
            op = Path(CFG.out_dir) / f"{p.stem}_overlay.jpg"
            print("overlay ->", draw_overlay(img, ev, op))

        if dump_json:
            out = Path(dump_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(ev.to_json())
            print("json ->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--no-rpn", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-attrs", action="store_true")
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--rpn-topk", type=int, default=None)
    ap.add_argument("--containment", type=float, default=None)
    ap.add_argument("--max-parent", type=float, default=None)
    ap.add_argument("--max-proposals", type=int, default=None)
    ap.add_argument("--attr-margin", type=float, default=None)
    a = ap.parse_args()

    if a.no_rpn:
        CFG.use_rpn = False
    if a.no_yolo:
        CFG.use_yolo = False
    if a.no_attrs:
        CFG.use_attributes = False
    if a.yolo:
        CFG.yolo_weights = a.yolo
    if a.conf is not None:
        CFG.yolo_conf = a.conf
    if a.rpn_topk is not None:
        CFG.rpn_pre_topk = a.rpn_topk
    if a.containment is not None:
        CFG.containment_thresh = a.containment
    if a.max_parent is not None:
        CFG.containment_max_parent = a.max_parent
    if a.max_proposals is not None:
        CFG.max_proposals = a.max_proposals
    if a.attr_margin is not None:
        CFG.attr_min_margin = a.attr_margin

    run(a.images, a.json, a.overlay)


if __name__ == "__main__":
    main()
