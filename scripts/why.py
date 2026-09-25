"""Everything about one image, in one paste-able block.

    python -m scripts.why data/eval/factory.jpg

    # match whatever the app is running
    JANA2_PROBES=1 JANA2_STUDENT=checkpoints_distill/projector_best.pt \
        python -m scripts.why data/eval/factory.jpg

WHY THIS EXISTS

A screenshot shows that an alert fired. It does not show WHICH MECHANISM
fired it, and probe failures and phrase failures need completely different
fixes:

    probe fired wrongly   -> the fitted vector is bad, or its AP was too low
    phrase fired wrongly  -> threshold is uncalibrated, or the phrasing is weak
    attribute fired       -> a softmax winner treated as evidence
    gated                 -> never ran at all; the requires= clause blocked it

So every check prints its score, its threshold, and the tag [probe] /
[phrase] / [gated]. Read that tag first.
"""
from __future__ import annotations
import sys

from PIL import Image

from jana2.config import CFG
from jana2.proposals import Proposer
from jana2.naming import Namer, build_evidence
from jana2 import readout
from jana2.postprocess import category


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/eval/factory.jpg"
    img = Image.open(path).convert("RGB")

    prop, namer = Proposer(), Namer()
    boxes, scores, srcs, labels, stats = prop(img)
    ev = build_evidence(path, img, boxes, scores, srcs, labels, namer)

    print(f"\n{'='*70}")
    print(f"{path}")
    print(f"probes={'ON' if getattr(CFG, 'use_probes', False) else 'OFF'}  "
          f"student={'ON' if getattr(CFG, 'clip_student_ckpt', None) else 'OFF'}  "
          f"clip={CFG.clip_model}")
    print("=" * 70)

    print("\n--- PROPOSALS")
    print(stats.render())

    print("\n--- OBJECTS (named_by: detector = label trusted, clip = zero-shot)")
    print(f"  {'id':<4}{'name':<26}{'category':<13}{'score':>6}"
          f"{'objness':>9}  {'src':<7}{'by':<9}")
    for o in ev.objects[:14]:
        sc = o.clip_names[0][1] if o.clip_names else 0.0
        print(f"  {o.idx:<4}{o.name[:25]:<26}{category(o.name):<13}"
              f"{sc:>6.2f}{o.objectness:>9.2f}  {o.source:<7}"
              f"{getattr(o, 'named_by', '?'):<9}")

    print("\n--- FRAME CHECKS (sorted by score)")
    print(f"  {'':<6}{'check':<20}{'score':>7}{'thr':>7}  mechanism")
    for a in sorted(ev.anomalies, key=lambda x: -x["prob"]):
        mark = "FIRED " if a["triggered"] else "      "
        mech = a.get("readout") or "gated"
        skip = a.get("skipped", "")
        print(f"  {mark}{a['name']:<20}{a['prob']:>7.3f}{a['threshold']:>7.2f}"
              f"  [{mech}] {skip}")

    print("\n--- PERSON CHECKS (only ones that fired are stored)")
    any_p = False
    for o in ev.objects:
        for k, v in (getattr(o, "checks", None) or {}).items():
            any_p = True
            print(f"  #{o.idx} {o.name[:16]:<18}{k:<18}{v['prob']:.3f}  "
                  f"({v['severity']})")
    if not any_p:
        print("  none fired")

    print("\n--- ATTRIBUTES")
    for o in ev.objects[:10]:
        if o.attributes:
            bits = " | ".join(f"{g}={v['label']} {v['prob']:.2f}"
                              for g, v in o.attributes.items())
            print(f"  #{o.idx} {o.name}: {bits}")

    print("\n--- THREAT")
    any_t = False
    for o in ev.objects:
        t = getattr(o, "threat", None)
        if t and t.get("level"):
            any_t = True
            print(f"  #{o.idx} {o.name}: {t['level']} / {t['tier']} "
                  f"— {t['reason']}")
    if not any_t:
        print("  none")

    sc = ev.scene or {}
    ok = sc.get("cos", 0.0) >= CFG.scene_min_cos
    print(f"\n--- SCENE  cos={sc.get('cos', 0.0):.3f} "
          f"(floor {CFG.scene_min_cos}) -> {'SPOKEN' if ok else 'suppressed'}")
    for n, p in (sc.get("clip_names") or [])[:4]:
        print(f"    {n:<40}{p:.3f}")

    print("\n--- RELATIONS")
    for r in (ev.relations or [])[:10]:
        print(f"  #{r['subject']} {r['subject_name']} --{r['predicate']}--> "
              f"#{r['object']} {r['object_name']}")
    if not ev.relations:
        print("  none")

    print("\n--- ALERTS")
    al = readout.alert_lines(ev)
    for x in al:
        print("  " + x)
    if not al:
        print("  none")

    print("\n--- CAPTION")
    try:
        from jana2.compose import compose
        print("  " + compose(ev))
    except Exception as e:                                    # noqa: BLE001
        print(f"  compose failed: {e}")

    print(f"\n--- TIMINGS ms: {ev.timings_ms}\n")


if __name__ == "__main__":
    main()
