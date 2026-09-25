"""Evidence -> text. No language model.

Two outputs:

  operator_lines()  one auditable line per entity, most important first
  caption()         plain prose

The rule that shapes both: a caption is about ENTITIES. Detections include
parts — a face, a sleeve, a chain. Those get attached to their owner
("a woman wearing a chain") and never listed beside it.

Nothing is stated that is not in the evidence. If little survives the
thresholds, say so rather than padding.
"""
from __future__ import annotations
from collections import defaultdict

from .config import CFG
from .postprocess import canon, category, NEVER_CAPTION
from . import threat

NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
       6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
_IRREGULAR = {"person": "people", "man": "men", "woman": "women",
              "child": "children"}
SEV_RANK = {"high": 0, "medium": 1, "low": 2}
ATTACHING = ("wearing", "has_part", "holding", "riding", "inside")

# A vehicle's own vehicle_state attribute is computed on a tight crop of just
# that vehicle — far more precise than the frame-level accident check, which
# has to find it again in a coarse whole-image tile alongside everything
# else in the scene. Treat a confident bad state as its own alert, same
# principle as person_checks catching a held weapon without ever needing a
# separate weapon detection.
VEHICLE_ALERT_STATES = {"damaged vehicle": "accident",
                        "overturned vehicle": "accident"}

# vehicle_state is a softmax across five options (parked / moving / damaged /
# overturned / open doors), so its winner is a RELATIVE ranking, not evidence.
# "damaged 44%" only means it beat second place by the attribute margin — on
# ordinary traffic the damaged option wins 0.44-0.58 by luck. A real wreck
# scores 0.88-0.92. Without this floor every car in a traffic jam was an
# accident alert.
VEHICLE_ALERT_MIN_PROB = 0.70

# States worth putting in a caption at all. "moving vehicle" and "parked
# vehicle" are what vehicles normally do — printing "(moving vehicle)" after
# every car on a road is noise that makes the real ones harder to spot.
NOTEWORTHY_STATES = {"damaged vehicle", "overturned vehicle",
                     "a vehicle with open doors", "vehicle with open doors",
                     "unattended bag on the ground", "a bag being placed down"}

# Checks that describe a person rather than report an incident. A guard in
# uniform is useful CONTEXT — it is not something an operator should be
# pulled out of their seat for. These still show on the object's own line,
# they just never reach the red banner. An alert list that cries wolf on
# "uniform 61%" trains people to ignore the banner entirely.
DESCRIPTIVE_CHECKS = {"uniform"}


def plural(word: str, n: int) -> str:
    if n == 1:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    head = word.split()[-1]
    if head in _IRREGULAR:
        return word[: -len(head)] + _IRREGULAR[head]
    if head.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if head.endswith("y") and head[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def count_phrase(name: str, n: int) -> str:
    if n == 1:
        return f"{'an' if name[0].lower() in 'aeiou' else 'a'} {name}"
    return f"{NUM.get(n, str(n))} {plural(name, n)}"


def join(items):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def where(o) -> str:
    return o.position.replace("-", " ")


# --------------------------------------------------------------------------- #
def attached_ids(evidence) -> set:
    """Objects spoken through their owner, so never listed on their own."""
    return {r["object"] for r in (evidence.relations or [])
            if r["predicate"] in ATTACHING}


def attachments(evidence, idx):
    """What this entity is wearing / holding / riding."""
    worn, held, rides = [], [], []
    for r in (evidence.relations or []):
        if r["subject"] != idx:
            continue
        if r["predicate"] == "wearing":
            worn.append(r["object_name"])
        elif r["predicate"] == "holding":
            held.append(r["object_name"])
        elif r["predicate"] in ("riding", "inside"):
            rides.append(r["object_name"])
    return worn, held, rides


def select(evidence, min_score=None, min_objectness=None, max_objects=8):
    min_score = CFG.caption_min_score if min_score is None else min_score
    min_objectness = (CFG.caption_min_objectness if min_objectness is None
                      else min_objectness)
    skip = attached_ids(evidence)

    objs = []
    for o in evidence.objects:
        if not o.clip_names or o.clip_names[0][1] < min_score:
            continue
        if o.objectness < min_objectness:
            continue
        if category(o.name) in NEVER_CAPTION:
            continue
        if o.idx in skip:
            continue
        objs.append(o)
    objs.sort(key=lambda o: -getattr(o, "importance", 0.0))
    return objs[:max_objects]


def person_words(o) -> tuple[list, str]:
    """(adjectives, headword) from the attribute pass."""
    a = o.attributes or {}
    bits = []
    if "age" in a:
        bits.append(a["age"]["label"].replace(" person", ""))
    if "build" in a:
        bits.append(a["build"]["label"].replace(" person", ""))
    head = a.get("gender", {}).get("label") or o.name
    return bits, head


def describe(evidence, o) -> str:
    """One entity, with everything attached to it folded in."""
    a = o.attributes or {}
    worn, held, rides = attachments(evidence, o.idx)

    if category(o.name) == "person":
        bits, head = person_words(o)
        s = " ".join(bits + [head])
        wear = list(worn)
        for g in ("upper_colour", "upper_type", "lower_colour", "lower_type"):
            if g in a:
                wear.append(a[g]["label"])
        if "headwear" in a:
            h = a["headwear"]["label"]
            if "uncovered" not in h and not h.startswith("not "):
                wear.append(h)
        if wear:
            s += " wearing " + join(wear[:3])
        carry = list(held)
        if "carrying" in a and a["carrying"]["label"] != "nothing":
            carry.append(a["carrying"]["label"])
        if carry:
            s += ", carrying " + join(carry[:2])
        if rides:
            s += ", on a " + rides[0]
        if "pose" in a and a["pose"]["label"] != "standing":
            s += f", {a['pose']['label']}"
        return s

    bits = [a[g]["label"].replace(" vehicle", "").replace(" bag", "")
            for g in ("vehicle_colour", "bag_colour") if g in a]
    name = (threat.phrase(o.name, o.threat) if category(o.name) == "weapon"
            and getattr(o, "threat", None) else o.name)
    s = " ".join(bits + [name])
    state = (a.get("vehicle_state", {}).get("label")
             or a.get("bag_state", {}).get("label"))
    if state in NOTEWORTHY_STATES:
        s += f" ({state})"
    return s


# --------------------------------------------------------------------------- #
def operator_lines(evidence, max_objects=8, min_score=None,
                   min_objectness=None) -> list[str]:
    lines = []
    for o in select(evidence, min_score, min_objectness, max_objects):
        flags = []
        for name, v in (getattr(o, "checks", None) or {}).items():
            mark = "!!" if v["severity"] == "high" else "!"
            flags.append(f"{mark} {name.replace('_', ' ')} {v['prob']:.2f}")
        t = getattr(o, "threat", None)
        if t and t.get("level"):
            mark = "!!" if t["severity"] == "high" else "!"
            flags.append(f"{mark} {t['level']} weapon — {t['reason']}")
        vs = (o.attributes or {}).get("vehicle_state")
        if (vs and vs["label"] in VEHICLE_ALERT_STATES
                and vs["prob"] >= VEHICLE_ALERT_MIN_PROB):
            flags.append(f"!! {VEHICLE_ALERT_STATES[vs['label']]} — "
                        f"{vs['label']} {vs['prob']:.2f}")
        n = getattr(o, "merged_count", 1)
        head = describe(evidence, o) + (f" (x{n} boxes)" if n > 1 else "")
        conf = o.clip_names[0][1] if o.clip_names else 0.0
        line = f"{head} · {where(o)} · {conf:.2f}"
        if flags:
            line += "   " + "  ".join(flags)
        lines.append(line)
    return lines


def alert_lines(evidence) -> list[str]:
    out = []
    for a in (evidence.anomalies or []):
        if a.get("triggered"):
            out.append((SEV_RANK.get(a["severity"], 1), -a["prob"],
                        f"{a['name'].replace('_', ' ')} {a['prob']:.0%} "
                        f"[{a['where']}]"))
    for o in evidence.objects:
        for name, v in (getattr(o, "checks", None) or {}).items():
            if name in DESCRIPTIVE_CHECKS:
                continue
            out.append((SEV_RANK.get(v["severity"], 1), -v["prob"],
                        f"{name.replace('_', ' ')} {v['prob']:.0%} "
                        f"[#{o.idx} {o.name}]"))
        t = getattr(o, "threat", None)
        if t and t.get("level"):
            out.append((SEV_RANK.get(t["severity"], 1), -1.0,
                        f"{t['level']} weapon: {t['reason']} "
                        f"[#{o.idx} {o.name}]"))
        vs = (o.attributes or {}).get("vehicle_state")
        if (vs and vs["label"] in VEHICLE_ALERT_STATES
                and vs["prob"] >= VEHICLE_ALERT_MIN_PROB):
            out.append((0, -vs["prob"],
                        f"{VEHICLE_ALERT_STATES[vs['label']]} "
                        f"({vs['label']}) {vs['prob']:.0%} [#{o.idx} {o.name}]"))
    out.sort()
    return [t[2] for t in out]


def caption(evidence, min_score=None, min_objectness=None,
            max_objects=8, detail=2) -> str:
    objs = select(evidence, min_score, min_objectness, max_objects)
    if not objs:
        return ("Nothing could be identified with enough confidence to "
                "describe. Lower the thresholds to see what was rejected.")

    scene = ""
    sc = (evidence.scene or {}).get("clip_names") or []
    if (detail >= 1 and len(sc) > 1
            and (sc[0][1] - sc[1][1]) > CFG.scene_min_margin
            and (evidence.scene or {}).get("cos", 0.0) >= CFG.scene_min_cos):
        scene = sc[0][0]

    groups = defaultdict(list)
    for o in objs:
        groups[canon(o.name)].append(o)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    def label(key, members):
        if len(members) == 1:
            return describe(evidence, members[0]) if detail >= 2 \
                else members[0].name
        names = {m.name for m in members}
        return names.pop() if len(names) == 1 else key

    listing = join([count_phrase(label(n, v), len(v)) for n, v in ordered])
    parts = [f"{scene[0].upper() + scene[1:]}. {listing.capitalize()}."
             if scene else f"The image shows {listing}."]

    if detail >= 2:
        said = 0
        for o in objs:
            if len(groups[canon(o.name)]) == 1:
                continue                      # already described in the list
            parts.append(f"In the {where(o)}, {describe(evidence, o)}.")
            said += 1
            if said >= 2:
                break

    alerts = alert_lines(evidence)
    if alerts:
        parts.insert(0, "ALERT: " + "; ".join(alerts[:3]) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
def object_rows(evidence, min_score=0.0):
    skip = attached_ids(evidence)
    rows = []
    for o in evidence.objects:
        if not o.clip_names or o.clip_names[0][1] < min_score:
            continue
        rows.append({
            "id": o.idx,
            "name": o.name,
            "importance": getattr(o, "importance", ""),
            "category": category(o.name),
            "spoken": "" if (o.idx in skip
                             or category(o.name) in NEVER_CAPTION) else "yes",
            "score": round(o.clip_names[0][1], 4),
            "margin": round(o.margin, 4),
            "objectness": round(o.objectness, 3),
            "boxes": getattr(o, "merged_count", 1),
            "source": o.source,
            "position": o.position,
            "alternatives": ", ".join(f"{n} {s:.3f}" for n, s in o.clip_names[1:4]),
            "flags": ", ".join(getattr(o, "checks", None) or {}),
            "attributes": "; ".join(f"{g}={v['label']}"
                                    for g, v in (o.attributes or {}).items()),
        })
    return rows


def rows_to_markdown(rows, cols=None) -> str:
    cols = cols or ["id", "name", "importance", "category", "spoken",
                    "score", "objectness", "boxes", "source", "position"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)
