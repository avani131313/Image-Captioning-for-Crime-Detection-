# Wiring the anomaly pass in — four small edits

## 1. `jana2/config.py` — add these fields inside `Config`

```python
    # ---- anomalies (third pass: yes/no over whole image + tiles) ----------
    use_anomalies: bool = True
    anomalies_txt: Path = ROOT / "assets" / "anomalies.txt"
    anomalies_path: Path = ROOT / "assets" / "anomalies.npz"
    anomaly_grids: tuple = (1, 2, 3)     # full frame, 2x2, 3x3 = 14 crops
```

## 2. `jana2/evidence.py` — one new field on `StructuredEvidence`

```python
    anomalies: list = field(default_factory=list)
```

Put it next to `relations`.

## 3. `jana2/naming.py`

In `Namer.__init__`, after the attributes block:

```python
        self.anom = None
        if cfg.use_anomalies:
            try:
                from .anomaly import AnomalyBank
                self.anom = AnomalyBank(cfg=cfg)
            except Exception as e:                            # noqa: BLE001
                print(f"[anom] disabled ({e})")
```

In `build_evidence`, just before the `return`:

```python
    t0 = time.perf_counter()
    anomalies = namer.anom.scan(image) if namer.anom is not None else []
    t["anomaly"] = (time.perf_counter() - t0) * 1000
```

and add `anomalies=anomalies,` to the `StructuredEvidence(...)` call.

## 4. `jana2/readout.py` — surface alerts in the caption

Add near the top:

```python
def alert_sentence(evidence) -> str:
    hits = [a for a in (evidence.anomalies or []) if a["triggered"]]
    if not hits:
        return ""
    hits.sort(key=lambda a: ({"high": 0, "medium": 1, "low": 2}[a["severity"]],
                             -a["prob"]))
    bits = [f"{a['name'].replace('_', ' ')} ({a['prob']:.0%})" for a in hits[:3]]
    return "ALERT: " + ", ".join(bits) + "."
```

and at the end of `caption()`, before the return:

```python
    alert = alert_sentence(evidence)
    if alert:
        parts.insert(0, alert)
```

## 5. `app.py` — a panel

```python
hits = [a for a in (ev.anomalies or []) if a["triggered"]]
if hits:
    st.error("  ".join(f"**{a['name']}** {a['prob']:.0%} ({a['where']})"
                       for a in hits))

with st.expander("anomaly checks — all scores"):
    st.markdown(readout.rows_to_markdown(
        [{"check": a["name"], "severity": a["severity"],
          "prob": a["prob"], "thr": a["threshold"],
          "where": a["where"], "hit": "YES" if a["triggered"] else ""}
         for a in ev.anomalies],
        cols=["check", "severity", "prob", "thr", "where", "hit"]))
```

## Build and run

```bash
cd /root/avani/code_files
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
.venv/bin/python -m scripts.build_anomalies
.venv/bin/streamlit run app.py --server.port 8610 --server.address 0.0.0.0
```

## Calibrate before trusting it

Open the "all scores" panel on ~20 **normal** frames from your cameras and
look at the highest probability each check reaches. Set each threshold a
little above that. Thresholds tuned on stock photos will not survive contact
with a real camera at night.

Cost: 14 crops through the image encoder, roughly 40-80 ms on the A100,
independent of how many objects were found.
