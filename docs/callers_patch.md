# Callers: Proposer now returns 5 values, build_evidence takes labels

`Proposer.__call__` → `(boxes, scores, srcs, labels, stats)`
`build_evidence(path, img, boxes, scores, srcs, labels, namer)`

Three call sites to update.

## 1. `jana2/evidence.py` — one new field on `ObjectSlot`

```python
    named_by: str = "clip"        # "yolo" (trusted label) or "clip"
```

Put it beside `source`. And in `markdown_table`, add it if you want it visible.

## 2. `scripts/probe.py`

```python
        boxes, scores, srcs, labels, stats = proposer(img)
        ...
        ev = build_evidence(p, img, boxes, scores, srcs, labels, namer)
```

## 3. `scripts/calibrate.py`

```python
        boxes, scores, srcs, labels, _ = proposer(img)
        ev = build_evidence(p, img, boxes, scores, srcs, labels, namer)
```

## 4. `app.py`

```python
boxes, scores, srcs, labels, stats = proposer(img)
ev = build_evidence("upload", img, boxes, scores, srcs, labels, namer)
```

Optionally show who named each object, in the sidebar or the table:

```python
CFG.detector_weights = st.selectbox(
    "detector", ["yolo26m.pt", "yolo26s.pt", "yolo26n.pt", "yolo11x.pt"], 0)
CFG.detector_imgsz = st.select_slider("detector imgsz", [640, 960, 1280, 1600], 1280)
CFG.yolo_trust_conf = st.slider("trust YOLO's label above", 0.1, 0.95, 0.45, 0.05)
```

Note `load_models` is `@st.cache_resource` — changing the detector needs it in
the cache key:

```python
@st.cache_resource(show_spinner="loading models…")
def load_models(detector: str, feature: str, use_rpn: bool, imgsz: int):
    CFG.detector_weights, CFG.feature_weights = detector, feature
    CFG.use_rpn, CFG.detector_imgsz = use_rpn, imgsz
    from jana2.proposals import Proposer
    from jana2.naming import Namer
    return Proposer(), Namer()
```

## Then

```bash
cd /root/avani/code_files
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
.venv/bin/python -m scripts.probe data/eval/factory.jpg --overlay
```

Expect on startup:

```
[detector] yolo26m.pt @ imgsz=1280
[feats] yolo26n.pt layers=[16, 19, 22] channels=[64, 128, 256] strides=[8, 16, 32]
[rpn] loaded ... recall 0.542
```

Two models now load. Detection runs at 1280 on a medium model; the nano runs
at 640 purely to feed rpn_v2. Roughly +60 ms per frame, and it should be a
large jump in what gets found.
