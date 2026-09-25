"""Streamlit UI — upload an image, get auditable evidence and a caption.

    .venv/bin/streamlit run app.py --server.port 8612 --server.address 0.0.0.0

Sidebar controls write to the REAL config fields. Anything that changes which
models load is part of the cache key, so switching a detector actually
reloads it instead of silently doing nothing.

NOTE: never st.dataframe / st.table on this host — pyarrow segfaults.
"""
from __future__ import annotations
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from jana2.config import CFG
from jana2 import readout
from jana2.postprocess import category, explain

st.set_page_config(page_title="JANA2 — surveillance readout", layout="wide")

CAT_COLOUR = {"weapon": (255, 40, 40), "person": (60, 200, 90),
              "vehicle": (80, 160, 255), "carried": (255, 190, 60),
              "animal": (200, 120, 255), "access": (150, 150, 150),
              "other": (180, 180, 180), "scenery": (90, 90, 90)}
SRC_DASH = {"yolo": 4, "world": 3, "yoloe": 3, "rpn": 2, "sam": 2}


@st.cache_resource(show_spinner="loading models…")
def load_models(sources: tuple, detector: str, feature: str, world: str,
                yoloe: str, imgsz: int):
    """Cache key must contain everything that changes which model loads."""
    CFG.sources = sources
    CFG.detector_weights = detector
    CFG.feature_weights = feature
    CFG.world_weights = world
    CFG.yoloe_weights = yoloe
    CFG.detector_imgsz = imgsz
    from jana2.proposals import Proposer
    from jana2.naming import Namer
    # Measured once per process — the cache returns this same value on every
    # rerun, so it reflects the real cold-start cost rather than a cache hit.
    t0 = time.perf_counter()
    p, n = Proposer(), Namer()
    return p, n, round(time.perf_counter() - t0, 1)


def overlay(img, ev, top_n=15):
    out = img.copy().convert("RGB")
    d = ImageDraw.Draw(out)
    for o in ev.objects[:top_n]:
        col = CAT_COLOUR.get(category(o.name), (200, 200, 200))
        w = 5 if getattr(o, "checks", None) else SRC_DASH.get(o.source, 3)
        d.rectangle(list(o.box), outline=col, width=w)
        lab = f"{o.idx}:{o.name} {o.importance:.2f} [{o.source}]"
        tb = d.textbbox((o.box[0], o.box[1]), lab)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=col)
        d.text((o.box[0], o.box[1]), lab, fill=(0, 0, 0))
    return out


# --------------------------------------------------------------------------- #
st.title("JANA2 — structured surveillance readout")

with st.sidebar:
    st.header("sources")
    st.caption(f"clip: {CFG.clip_model}")
    sources = st.multiselect(
        "proposal sources (order = priority)",
        ["yolo", "world", "yoloe"], default=list(CFG.sources),
        help="yolo = COCO 80 classes, yolo26n, trusted labels · "
             "world = YOLO-World open-vocab · yoloe = YOLOE open-vocab, "
             "newer. world and yoloe share assets/world_classes.txt, so "
             "you can run one against the other on the same list.")

    DETECTOR_OPTS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"]
    detector = st.selectbox(
        "COCO detector", DETECTOR_OPTS,
        index=DETECTOR_OPTS.index(CFG.detector_weights)
        if CFG.detector_weights in DETECTOR_OPTS else 0)

    WORLD_OPTS = ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt",
                  "yolov8l-worldv2.pt", "yolov8x-worldv2.pt"]
    world_w = st.selectbox(
        "world model", WORLD_OPTS,
        index=WORLD_OPTS.index(CFG.world_weights)
        if CFG.world_weights in WORLD_OPTS else 0)

    YOLOE_OPTS = ["yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt",
                  "yoloe-11s-seg.pt", "yoloe-11m-seg.pt"]
    yoloe_w = st.selectbox(
        "yoloe model", YOLOE_OPTS,
        index=YOLOE_OPTS.index(CFG.yoloe_weights)
        if CFG.yoloe_weights in YOLOE_OPTS else 0)

    imgsz = st.select_slider("detector imgsz", [640, 960, 1280, 1600],
                             value=CFG.detector_imgsz)

    st.header("thresholds")
    CFG.yolo_conf = st.slider("yolo conf", 0.01, 0.9, CFG.yolo_conf, 0.01)
    CFG.yolo_trust_conf = st.slider("trust yolo label above", 0.05, 0.95,
                                    CFG.yolo_trust_conf, 0.05)
    CFG.world_conf = st.slider("world conf", 0.01, 0.9, CFG.world_conf, 0.01)
    CFG.world_trust_conf = st.slider("trust world label above", 0.02, 0.9,
                                     CFG.world_trust_conf, 0.01)
    CFG.yoloe_conf = st.slider("yoloe conf", 0.01, 0.9, CFG.yoloe_conf, 0.01)
    CFG.yoloe_trust_conf = st.slider("trust yoloe label above", 0.02, 0.9,
                                     CFG.yoloe_trust_conf, 0.01)

    st.header("cleanup")
    CFG.nms_iou = st.slider("nms iou", 0.1, 0.95, CFG.nms_iou, 0.05)
    CFG.containment_min_child_ratio = st.slider(
        "part-merge child ratio", 0.05, 1.0,
        CFG.containment_min_child_ratio, 0.05,
        help="lower keeps more small objects — a held knife survives")
    CFG.dedup_iou = st.slider("merge iou", 0.05, 0.95, CFG.dedup_iou, 0.05)
    CFG.min_area_frac = st.slider("min area frac", 0.0001, 0.02,
                                  CFG.min_area_frac, 0.0001, format="%.4f")
    CFG.max_proposals = st.slider("max proposals", 4, 96, CFG.max_proposals)

    st.header("report")
    top_n = st.slider("objects to report", 1, 30, 10)
    min_score = st.slider("min name score", 0.0, 0.5,
                          CFG.caption_min_score, 0.005)
    min_obj = st.slider("min objectness", 0.0, 0.9,
                        CFG.caption_min_objectness, 0.01)
    CFG.attr_min_margin = st.slider("attribute margin", 0.0, 0.9,
                                    CFG.attr_min_margin, 0.01)

up = st.file_uploader("image", type=["jpg", "jpeg", "png", "bmp", "webp"])
path = st.text_input("…or a path on the server", "data/eval/factory.jpg")

img = None
if up is not None:
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
elif path and Path(path).exists():
    img = Image.open(path).convert("RGB")
if img is None:
    st.info("upload an image or give a path that exists on the server")
    st.stop()

if not sources:
    st.error("pick at least one proposal source")
    st.stop()

proposer, namer, load_s = load_models(tuple(sources), detector,
                                      CFG.feature_weights, world_w, yoloe_w,
                                      imgsz)

from jana2.naming import build_evidence
t0 = time.perf_counter()
boxes, scores, srcs, labels, stats = proposer(img)
ev = build_evidence("upload", img, boxes, scores, srcs, labels, namer)
ev.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)

# what is actually live — no more guessing
live = []
live.append(f"sources={'+'.join(sources)}")
live.append(f"world={'ON' if proposer.world is not None else 'OFF'}")
live.append(f"yoloe={'ON' if proposer.yoloe is not None else 'OFF'}")
live.append(f"rpn={'ON' if proposer.rpn is not None else 'OFF'}")
live.append(f"attrs={'ON' if namer.attrs is not None else 'OFF'}")
live.append(f"person={'ON' if namer.person_checks is not None else 'OFF'}")
live.append(f"anomaly={'ON' if namer.frame_checks is not None else 'OFF'}")
live.append(f"scene={'ON' if namer.scene_t is not None else 'OFF'}")
live.append(f"probes={'ON' if getattr(CFG, 'use_probes', False) else 'OFF'}")
live.append(f"clip={'STUDENT' if getattr(CFG, 'clip_student_ckpt', None) else 'teacher'}")
st.caption(" · ".join(live))

# ---- runtime -------------------------------------------------------- #
t = ev.timings_ms
c1, c2, c3, c4 = st.columns(4)
c1.metric("this image", f"{t.get('total', 0):.0f} ms")
c2.metric("detection", f"{t.get('total', 0) - sum(v for k, v in t.items() if k != 'total'):.0f} ms",
          help="total minus the CLIP stages below")
c3.metric("clip + naming",
          f"{t.get('clip_roi', 0) + t.get('naming', 0):.0f} ms")
c4.metric("model load", f"{load_s:.1f} s",
          help="one-off cold start, cached for the process")
st.caption("stage breakdown: " + " · ".join(
    f"{k} {v:.0f}ms" for k, v in sorted(t.items(), key=lambda kv: -kv[1])))

alerts = readout.alert_lines(ev)
if alerts:
    st.error("**ALERTS**  ·  " + "   ·   ".join(alerts[:6]))

left, right = st.columns([1, 1])
with left:
    st.image(overlay(img, ev, top_n), use_container_width=True)
    st.caption("red weapon · green person · blue vehicle · amber carried · "
               "grey other — thick border = a check fired")
with right:
    st.subheader("operator view")
    lines = readout.operator_lines(ev, top_n, min_score, min_obj)
    st.code("\n".join(lines) if lines else "nothing above thresholds",
            language=None)
    st.subheader("caption")
    st.info(readout.caption(ev, min_score, min_obj, top_n))

st.subheader("proposal funnel")
st.code(stats.render())

rows = readout.object_rows(ev, min_score)
st.subheader(f"objects ({len(rows)} after merging)")
st.markdown(readout.rows_to_markdown(
    rows, cols=["id", "name", "importance", "category", "score",
                "objectness", "boxes", "source", "position", "flags"]))

with st.expander("relations"):
    if ev.relations:
        st.markdown(readout.rows_to_markdown(
            [{"subject": f"#{r['subject']} {r['subject_name']}",
              "predicate": r["predicate"],
              "object": f"#{r['object']} {r['object_name']}",
              "conf": r["confidence"], "why": str(r["evidence"])}
             for r in ev.relations],
            cols=["subject", "predicate", "object", "conf", "why"]))
    else:
        st.write("none — geometry was not clear enough to state one")

with st.expander("why this ranking?"):
    st.markdown(readout.rows_to_markdown(
        [dict(id=o.idx, obj=o.name, **explain(o, ev.width, ev.height))
         for o in ev.objects[:top_n]],
        cols=["id", "obj", "category", "cat_weight", "area", "center",
              "objectness", "name_conf", "total"]))

with st.expander("attributes"):
    any_a = False
    for o in ev.objects[:top_n]:
        if o.attributes:
            any_a = True
            st.markdown(f"**#{o.idx} {o.name}** — " + " · ".join(
                f"{g}: {v['label']} ({v['prob']:.2f})"
                for g, v in o.attributes.items()))
    if not any_a:
        st.write("none passed the margin test — lower 'attribute margin'")

with st.expander("scene + anomaly scores"):
    _cos = (ev.scene or {}).get("cos", 0.0)
    st.write(f"scene raw cos **{_cos:.4f}** "
             f"(floor {CFG.scene_min_cos:.2f} — below this the caption says "
             f"nothing about the location)")
    st.write("scene:", (ev.scene or {}).get("clip_names", [])[:5])
    if ev.anomalies:
        st.markdown(readout.rows_to_markdown(
            [{"check": a["name"], "sev": a["severity"], "prob": a["prob"],
              "thr": a["threshold"], "where": a["where"],
              "hit": "YES" if a["triggered"] else ""} for a in ev.anomalies],
            cols=["check", "sev", "prob", "thr", "where", "hit"]))

with st.expander("alternatives / timings / json"):
    for r in rows:
        st.markdown(f"**#{r['id']} {r['name']}** ({r['score']}) → "
                    f"{r['alternatives'] or '—'}")
    st.write(ev.timings_ms)
    st.code(ev.to_json()[:8000])
