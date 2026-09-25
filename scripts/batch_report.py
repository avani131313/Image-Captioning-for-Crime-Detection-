"""Run the pipeline over a folder and produce one scannable HTML page.

    python -m scripts.batch_report --images data/eval --out outputs/report.html
    python -m scripts.batch_report --images /path/to/frames --limit 40

Clicking through Streamlit one image at a time hides the pattern. Twenty
captions on one page makes it obvious whether a failure is a one-off or
systematic — and which stage is causing it.

The page is self-contained (images inlined), so you can scp it and open it
locally.
"""
from __future__ import annotations
import argparse
import base64
import html
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw

from jana2.config import CFG
from jana2 import readout, compose
from jana2.postprocess import category

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAT_COLOUR = {"weapon": "#ff2828", "person": "#3cc85a", "vehicle": "#50a0ff",
              "carried": "#ffbe3c", "animal": "#c878ff", "access": "#969696",
              "other": "#b4b4b4", "scenery": "#5a5a5a",
              "garment": "#ff9ecb", "accessory": "#ff9ecb", "part": "#404040"}


def images(folder, limit=None):
    fs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG)
    return fs[:limit] if limit else fs


def overlay_b64(img, ev, top_n=12, width=560):
    out = img.copy().convert("RGB")
    d = ImageDraw.Draw(out)
    for o in ev.objects[:top_n]:
        col = CAT_COLOUR.get(category(o.name), "#b4b4b4")
        d.rectangle(list(o.box), outline=col, width=max(2, out.width // 400))
        lab = f"{o.idx}:{o.name}"
        tb = d.textbbox((o.box[0], o.box[1]), lab)
        d.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=col)
        d.text((o.box[0], o.box[1]), lab, fill="#000000")
    if out.width > width:
        out = out.resize((width, int(out.height * width / out.width)))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


CSS = """
body{background:#0f1116;color:#e6e6e6;font:15px/1.55 -apple-system,
 BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8b93a7;margin-bottom:24px}
.card{display:grid;grid-template-columns:580px 1fr;gap:22px;
 border-top:1px solid #262b36;padding:22px 0}
img{border-radius:8px;width:100%}
.cap{background:#16203a;border-left:3px solid #4a86ff;padding:12px 14px;
 border-radius:6px;margin-bottom:12px;font-size:16px}
.alert{background:#3a1616;border-left:3px solid #ff4a4a;padding:10px 14px;
 border-radius:6px;margin-bottom:12px;color:#ffb3b3}
pre{background:#161a22;padding:11px 13px;border-radius:6px;overflow-x:auto;
 font:12px/1.5 ui-monospace,Menlo,monospace;color:#c8d0e0;margin:0 0 10px}
.name{color:#8b93a7;font-size:13px;margin-bottom:8px}
details{margin-top:8px} summary{cursor:pointer;color:#8b93a7;font-size:13px}
.tag{display:inline-block;background:#1d222d;border-radius:4px;padding:1px 7px;
 margin-right:5px;font-size:12px;color:#9aa3b5}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/report.html")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    from jana2.proposals import Proposer
    from jana2.naming import Namer, build_evidence

    paths = images(a.images, a.limit)
    if not paths:
        raise SystemExit(f"no images under {a.images}")
    print(f"[report] {len(paths)} images")
    proposer, namer = Proposer(), Namer()

    cards, t_all = [], time.perf_counter()
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:                                # noqa: BLE001
            print(f"  skip {p.name}: {e}")
            continue
        t0 = time.perf_counter()
        boxes, scores, srcs, labels, stats = proposer(img)
        ev = build_evidence(p, img, boxes, scores, srcs, labels, namer)
        ms = (time.perf_counter() - t0) * 1000

        cap = compose.compose(ev)
        alerts = readout.alert_lines(ev)
        ops = readout.operator_lines(ev, a.top)
        rels = [f"{r['subject_name']} {r['predicate']} {r['object_name']}"
                for r in (ev.relations or [])]
        sc = (ev.scene or {}).get("clip_names") or []

        cards.append(f"""
<div class="card">
  <div>
    <img src="data:image/jpeg;base64,{overlay_b64(img, ev, a.top)}">
    <div class="name">{html.escape(p.name)} · {ev.width}x{ev.height} ·
      {ms:.0f} ms · {stats.final} boxes {html.escape(str(stats.by_source_final))}</div>
  </div>
  <div>
    {'<div class="alert"><b>ALERT</b> ' + html.escape('; '.join(alerts[:4])) + '</div>' if alerts else ''}
    <div class="cap">{html.escape(cap)}</div>
    <div>{''.join(f'<span class="tag">{html.escape(n)} {s:.2f}</span>' for n, s in sc[:3])}</div>
    <pre>{html.escape(chr(10).join(ops) or 'nothing above thresholds')}</pre>
    <details><summary>relations ({len(rels)})</summary>
      <pre>{html.escape(chr(10).join(rels) or 'none')}</pre></details>
    <details><summary>funnel</summary><pre>{html.escape(stats.render())}</pre></details>
  </div>
</div>""")
        print(f"  {i}/{len(paths)} {p.name}  {ms:.0f}ms", flush=True)

    total = time.perf_counter() - t_all
    page = f"""<!doctype html><meta charset="utf-8">
<title>JANA2 batch report</title><style>{CSS}</style>
<h1>JANA2 — {len(cards)} images</h1>
<div class="sub">{html.escape('+'.join(CFG.sources))} ·
 {html.escape(CFG.detector_weights)} @ {CFG.detector_imgsz} ·
 {html.escape(CFG.clip_model)} · {total:.1f}s total
 ({total / max(len(cards), 1) * 1000:.0f} ms/image)</div>
{''.join(cards)}"""

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"\n[report] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("view it with:  python -m http.server 8615 --directory outputs")


if __name__ == "__main__":
    main()
