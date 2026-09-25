"""Relations between objects, from geometry alone.

"A man riding a motorcycle" is not a harder perception problem than "a man"
and "a motorcycle" — it is those two boxes in a particular arrangement.

Relations are load-bearing here: the caption uses them to attach garments,
accessories and body parts to their owner instead of listing them as peers.
A face that produces a has_part relation disappears from the sentence; a
face that produces none gets listed beside the person.

Every rule is strict. A relation that fires wrongly is a sentence stating
something that never happened, so when geometry is ambiguous we emit nothing.
"""
from __future__ import annotations
from .postprocess import category, canon, GARMENT, ACCESSORY

WORN = GARMENT | ACCESSORY
TWO_WHEELER = {"motorcycle", "scooter", "bicycle"}
ENCLOSED = {"car", "bus", "truck", "van", "auto rickshaw", "jeep",
            "ambulance", "tractor"}
HELD = {"bag", "backpack", "handbag", "suitcase", "trolley bag", "box",
        "carton", "cardboard box", "parcel", "sack", "bottle",
        "water bottle", "mobile phone", "umbrella", "laptop", "cylinder",
        "gas cylinder", "shopping bag", "plastic bag", "cash", "jewellery",
        "food delivery bag", "knife", "gun", "stick", "weapon"}
STATIC = {"gate", "main gate", "society gate", "boom barrier", "door",
          "glass door", "turnstile", "metal detector", "billing counter",
          "cash counter", "reception desk", "atm machine", "shutter",
          "security cabin", "guard booth", "entrance gate", "exit gate"}


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _inside_frac(child, parent):
    a = _area(child)
    return _inter(child, parent) / a if a > 0 else 0.0


def _h_overlap(a, b):
    o = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return o / max(min(a[2] - a[0], b[2] - b[0]), 1e-9)


def _v_overlap(a, b):
    o = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return o / max(min(a[3] - a[1], b[3] - b[1]), 1e-9)


def _gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _rel(s, p, o, conf, **ev):
    return {"subject": s.idx, "subject_name": s.name, "predicate": p,
            "object": o.idx, "object_name": o.name,
            "confidence": round(conf, 3), "evidence": ev}


# --------------------------------------------------------------------------- #
def derive(objects, W, H, max_relations=16) -> list[dict]:
    out = []
    diag = (W * W + H * H) ** 0.5

    for a in objects:
        ca, na = category(a.name), canon(a.name)
        for b in objects:
            if a.idx == b.idx:
                continue
            cb, nb = category(b.name), canon(b.name)
            ab, bb = a.box, b.box

            # ---- body part of a person --------------------------------
            if cb == "person" and ca == "part":
                f = _inside_frac(ab, bb)
                if f > 0.6:
                    out.append(_rel(b, "has_part", a, 0.90, inside=round(f, 2)))
                    continue

            # ---- worn by a person -------------------------------------
            # No position rule: a salwar kameez covers the whole body, a cap
            # only the top. Containment is the reliable signal.
            if cb == "person" and na in WORN:
                f = _inside_frac(ab, bb)
                if f > 0.55:
                    out.append(_rel(b, "wearing", a, 0.80, inside=round(f, 2)))
                    continue

            # ---- person riding a two-wheeler --------------------------
            if ca == "person" and nb in TWO_WHEELER:
                if _h_overlap(ab, bb) > 0.5 and _v_overlap(ab, bb) > 0.15:
                    out.append(_rel(a, "riding", b, 0.85,
                                    h_overlap=round(_h_overlap(ab, bb), 2)))
                    continue

            # ---- person inside a car/bus ------------------------------
            if ca == "person" and nb in ENCLOSED:
                f = _inside_frac(ab, bb)
                if f > 0.75:
                    out.append(_rel(a, "inside", b, 0.75, inside=round(f, 2)))
                    continue

            # ---- held by a person -------------------------------------
            if cb == "person" and (na in HELD or ca == "weapon" or ca == "carried"):
                f = _inside_frac(ab, bb)
                ratio = _area(ab) / max(_area(bb), 1e-9)
                if f > 0.55 and ratio < 0.35:
                    conf = 0.85 if ca == "weapon" else 0.70
                    out.append(_rel(b, "holding", a, conf,
                                    inside=round(f, 2),
                                    size_ratio=round(ratio, 3)))
                    continue

            # ---- person at a gate / counter ---------------------------
            if ca == "person" and nb in STATIC:
                g = _gap(ab, bb) / diag
                if g < 0.03:
                    out.append(_rel(a, "at", b, 0.65, gap=round(g, 3)))
                    continue

    # ---- side by side, only among the important few ------------------
    top = objects[:6]
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            if category(a.name) in ("scenery", "part", "garment", "accessory"):
                continue
            if category(b.name) in ("scenery", "part", "garment", "accessory"):
                continue
            if any({r["subject"], r["object"]} == {a.idx, b.idx} for r in out):
                continue
            g = _gap(a.box, b.box) / diag
            if g < 0.02 and _v_overlap(a.box, b.box) > 0.45:
                out.append(_rel(a, "next to", b, 0.55, gap=round(g, 3)))

    # one relation per (subject, predicate, object); confident ones first
    out.sort(key=lambda r: -r["confidence"])
    seen, kept = set(), []
    for r in out:
        key = (r["subject"], r["predicate"], r["object"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept[:max_relations]


def phrase(rel) -> str:
    p = rel["predicate"]
    s, o = rel["subject_name"], rel["object_name"]
    if p == "wearing":
        return f"{s} wearing a {o}"
    if p == "holding":
        return f"{s} holding a {o}"
    if p == "riding":
        return f"{s} riding a {o}"
    if p == "inside":
        return f"{s} inside a {o}"
    if p == "at":
        return f"{s} at the {o}"
    if p == "has_part":
        return f"{s} with a visible {o}"
    return f"{s} {p} a {o}"
