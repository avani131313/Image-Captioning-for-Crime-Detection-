# Wiring relations in — two small edits

## `jana2/naming.py` — inside `build_evidence`, after the ranking line

Find:

```python
    objs = rank(objs, W, H, hot)
```

Add straight after it:

```python
    t0 = time.perf_counter()
    rels = []
    if CFG.use_relations:
        from .relations import derive
        rels = derive(objs, W, H, CFG.max_relations)
    t["relations"] = (time.perf_counter() - t0) * 1000
```

and pass `relations=rels,` into the `StructuredEvidence(...)` call.

Relations must be derived AFTER `rank()`, because the rules reference
`o.idx` and ranking renumbers everything.

## `jana2/readout.py` — use them in the caption

Add near the top:

```python
from .relations import phrase as rel_phrase
```

Then inside `caption()`, replace the attribute-sentence block with this — it
prefers a relation over a bare attribute list, because "a man riding a
motorcycle" says more than "a man" plus "a motorcycle":

```python
    if detail >= 2:
        used, said = set(), 0
        for r in (evidence.relations or []):
            if r["confidence"] < 0.6:
                continue
            parts.append(f"There is {rel_phrase(r)} in the "
                         f"{_pos_of(evidence, r['subject'])} of the frame.")
            used.add(r["subject"]); used.add(r["object"])
            said += 1
            if said >= 3:
                break
        for o in objs:
            if said >= 4 or o.idx in used:
                continue
            bits = (_person_bits(o) if category(o.name) == "person"
                    else _object_bits(o))
            if not bits:
                continue
            parts.append(f"In the {where(o)}, {join(bits[:4])}.")
            said += 1
```

and a small helper beside `where()`:

```python
def _pos_of(evidence, idx):
    for o in evidence.objects:
        if o.idx == idx:
            return where(o)
    return "frame"
```

## Also show them in the UI — `app.py`

```python
with st.expander("relations"):
    if ev.relations:
        st.markdown(readout.rows_to_markdown(
            [{"subject": f"#{r['subject']} {r['subject_name']}",
              "predicate": r["predicate"],
              "object": f"#{r['object']} {r['object_name']}",
              "conf": r["confidence"],
              "evidence": str(r["evidence"])} for r in ev.relations],
            cols=["subject", "predicate", "object", "conf", "evidence"]))
    else:
        st.write("none — geometry was not clear enough to state one")
```

## Build and run

```bash
cd /root/avani/code_files
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
.venv/bin/streamlit run app.py --server.port 8610 --server.address 0.0.0.0
```

First run downloads `yolov8x-worldv2.pt` (~150 MB).

Expect at startup:

```
[yolo]   yolo26m.pt @ imgsz=1280
[world]  yolov8x-worldv2.pt with 140 classes
[feats]  yolo26n.pt channels=[64, 128, 256] strides=[8, 16, 32]
[rpn]    loaded ... recall 0.542
```

## Turning rpn_v2 off

If YOLO-World covers your objects well, drop it:

```python
sources: tuple = ("yolo", "world")
```

Compare the `by_source_final` counts with and without. If `rpn` contributes
almost nothing once World is running, it has been replaced and you can stop
carrying it.
