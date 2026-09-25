# Wiring the gated checks — three small edits

## 1. `jana2/config.py` — add inside `Config`

```python
    # ---- yes/no check scoring ---------------------------------------------
    # CLIP's own logit_scale (~100) turns a 0.05 cosine gap into 99%
    # confidence. That is calibrated for retrieval across a batch, not for a
    # 2-way comparison. Use a small temperature so the number means something.
    check_temperature: float = 20.0
    # and a check must clear a RAW cosine margin too — preferring one bad
    # match over another bad match is not evidence.
    check_min_cos_margin: float = 0.02
```

Raise `check_temperature` for sharper decisions, lower it for softer ones.
Raise `check_min_cos_margin` to kill more false alarms.

## 2. `jana2/naming.py` — pass context so gating can work

In `build_evidence`, replace the anomaly block with:

```python
    t0 = time.perf_counter()
    ctx = {
        "names": {(o.name or "").lower() for o in objs}
                 | {canon(o.name) for o in objs},
        "n_people": sum(1 for o in objs if category(o.name) == "person"),
    }
    anomalies = (namer.frame_checks.scan_frame(image, ctx)
                 if namer.frame_checks is not None else [])
    t["anomaly"] = (time.perf_counter() - t0) * 1000
```

and add `canon` to the import at the top:

```python
from .postprocess import merge_duplicates, rank, category, canon
```

This is the important one. Without the context dict, `requires=` can't gate
anything and you get `no helmet 99%` on a portrait.

## 3. `app.py` — show gated checks separately

In the anomaly expander:

```python
        st.markdown(readout.rows_to_markdown(
            [{"check": a["name"], "sev": a["severity"], "prob": a["prob"],
              "cos": a.get("cos_margin", ""), "thr": a["threshold"],
              "where": a["where"],
              "hit": "YES" if a["triggered"] else (a.get("skipped") or "")}
             for a in ev.anomalies],
            cols=["check", "sev", "prob", "cos", "thr", "where", "hit"]))
```

Gated checks now show `requirement not met` instead of a fake score, so you
can see the gate working.

## Rebuild and run

```bash
cd /root/avani/code_files
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
.venv/bin/python -m scripts.build_checks
.venv/bin/streamlit run app.py --server.port 8612 --server.address 0.0.0.0
```

On the portrait, expect: `no_helmet`, `wrong_side`, `accident`,
`overloaded_vehicle`, `unattended_bag`, `crowd_surge`, `fight`,
`loitering_group` all **gated** — no vehicle, no bag, one person. `darkness`
and `flooding` scored on the full frame only, so a dark jacket no longer
counts as a dark scene.

If anything still fires, raise `check_min_cos_margin` to 0.04 and look at the
`cos` column — that number is the honest one. The probability is derived from
it, so if `cos` is near zero the model is telling you it cannot distinguish
the two statements, whatever the percentage says.

## Still off: scene

`scene=OFF` in your status line. `assets/scenes.npz` never built:

```bash
.venv/bin/python -m scripts.build_vocab --names-from assets/scenes.txt --out assets/scenes.npz
```
