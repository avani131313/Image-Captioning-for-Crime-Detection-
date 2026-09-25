"""Assemble evidence into a caption a person would actually write.

The evidence is already correct by this point. What was missing was
composition — the difference between

    The image shows a person, a car, a car and a person.

and

    A busy city street. In the foreground a young man in a dark shirt stands
    between two parked cars, holding a long blade. Several people are visible
    further back.

Everything here is deterministic. No model, no sampling, so any wrong word
traces to a rule you can read.

Four ideas do most of the work:

  DEPTH      small and high in the frame means far away. "Several people in
             the background" beats "eight people" for a wide street shot.
  SUBJECT    one entity leads the description; everything else is context.
  GROUPING   three cars are "three cars", not three sentences.
  VERBS      chosen from pose and relations, so sentences stop all starting
             "There is".
"""
from __future__ import annotations
from collections import defaultdict

from .config import CFG
from .postprocess import canon, category, NEVER_CAPTION
from .readout import (attached_ids, attachments, join, count_phrase,
                      plural, alert_lines)
from . import threat

# --------------------------------------------------------------------------- #
FOREGROUND, MIDGROUND, BACKGROUND = "fg", "mg", "bg"

SPATIAL = {
    "top-left": "in the upper left", "top-center": "at the top",
    "top-right": "in the upper right", "middle-left": "on the left",
    "center": "in the centre", "middle-right": "on the right",
    "bottom-left": "in the lower left", "bottom-center": "in the foreground",
    "bottom-right": "in the lower right",
}

STATIC_VERB = {"vehicle": "is parked", "access": "stands",
               "carried": "sits", "animal": "is"}


def prominence(o) -> str:
    """Rough depth from size and height in frame. Cameras look outward and
    down, so small things high in the image are far away."""
    if o.area_frac >= 0.12:
        return FOREGROUND
    cy = (o.box[1] + o.box[3]) / 2
    high = cy < 0.55 * max(getattr(o, "_frame_h", 1), 1)
    if o.area_frac <= 0.015 and high:
        return BACKGROUND
    if o.area_frac >= 0.04:
        return FOREGROUND
    return MIDGROUND


def spatial(o, depth=None) -> str:
    if depth == BACKGROUND:
        return "in the background"
    return SPATIAL.get(o.position, "in the frame")


def verb_for(evidence, o) -> str:
    a = o.attributes or {}
    _, _, rides = attachments(evidence, o.idx)
    cat = category(o.name)
    if cat == "person":
        if rides:
            return "is riding"
        pose = a.get("pose", {}).get("label")
        if pose == "walking":
            return "is walking"
        if pose == "running":
            return "is running"
        if pose == "sitting":
            return "is sitting"
        if pose in ("lying on the ground", "collapsed"):
            return "is lying on the ground"
        if pose == "squatting":
            return "is crouching"
        return "stands"
    return STATIC_VERB.get(cat, "is")


# --------------------------------------------------------------------------- #
def noun_phrase(evidence, o, detailed=True, article=True) -> str:
    """The entity plus whatever is attached to it."""
    a = o.attributes or {}
    worn, held, rides = attachments(evidence, o.idx)
    cat = category(o.name)

    if cat == "person":
        adj = []
        if detailed and "age" in a:
            adj.append(a["age"]["label"].replace(" person", ""))
        head = a.get("gender", {}).get("label") or o.name
        wear = list(worn)
        if detailed:
            for g in ("upper_colour", "upper_type"):
                if g in a:
                    wear.append(a[g]["label"])
            if "headwear" in a:
                h = a["headwear"]["label"]
                if "uncovered" not in h and not h.startswith("not "):
                    wear.append(h)
        s = " ".join(adj + [head])
        if wear:
            s += " in " + join(wear[:2]) if len(wear) == 1 else \
                 " wearing " + join(wear[:2])
    else:
        adj = []
        if detailed:
            for g in ("vehicle_colour", "bag_colour"):
                if g in a:
                    adj.append(a[g]["label"].replace(" vehicle", "")
                               .replace(" bag", ""))
        name = (threat.phrase(o.name, o.threat) if category(o.name) == "weapon"
                and getattr(o, "threat", None) else o.name)
        s = " ".join(adj + [name])

    if article:
        s = f"{'an' if s[0].lower() in 'aeiou' else 'a'} {s}"
    return s


def trailing_clause(evidence, o) -> str:
    """Held items, mounts and flagged checks — the part after the comma."""
    a = o.attributes or {}
    worn, held, rides = attachments(evidence, o.idx)
    bits = []

    carry = list(held)
    if "carrying" in a and a["carrying"]["label"] != "nothing":
        carry.append(a["carrying"]["label"])
    if carry:
        bits.append("holding " + join(carry[:2]))
    if rides:
        bits.append("on a " + rides[0])

    for name, v in (getattr(o, "checks", None) or {}).items():
        if v["severity"] == "high" and name not in ("uniform",):
            bits.append(name.replace("_", " "))
    return join(bits)


# --------------------------------------------------------------------------- #
def compose(evidence, min_score=None, min_objectness=None,
            max_entities=10, detail=2) -> str:
    min_score = CFG.caption_min_score if min_score is None else min_score
    min_objectness = (CFG.caption_min_objectness if min_objectness is None
                      else min_objectness)

    skip = attached_ids(evidence)
    ents = []
    for o in evidence.objects:
        if not o.clip_names or o.clip_names[0][1] < min_score:
            continue
        if o.objectness < min_objectness:
            continue
        if category(o.name) in NEVER_CAPTION or o.idx in skip:
            continue
        o._frame_h = evidence.height
        ents.append(o)

    if not ents:
        return ("Nothing in this image could be identified with enough "
                "confidence to describe.")

    ents.sort(key=lambda o: -getattr(o, "importance", 0.0))
    ents = ents[:max_entities]
    for o in ents:
        o._depth = prominence(o)

    sentences = []

    # ---- 1. scene ----------------------------------------------------
    sc = (evidence.scene or {}).get("clip_names") or []
    scene = ""
    if (len(sc) > 1
            and (sc[0][1] - sc[1][1]) > CFG.scene_min_margin
            and (evidence.scene or {}).get("cos", 0.0) >= CFG.scene_min_cos):
        scene = sc[0][0]
        sentences.append(scene[0].upper() + scene[1:] + ".")

    # ---- 2. the subject ----------------------------------------------
    subj = ents[0]
    near = [o for o in ents[1:] if o._depth == FOREGROUND]
    others = [o for o in ents[1:] if o._depth != FOREGROUND]

    s = noun_phrase(evidence, subj, detail >= 2).capitalize()
    s += " " + verb_for(evidence, subj) + " " + spatial(subj, subj._depth)
    tail = trailing_clause(evidence, subj) if detail >= 2 else ""
    if tail:
        s += ", " + tail
    sentences.append(s + ".")

    # ---- 3. other foreground entities, grouped -----------------------
    if near:
        groups = defaultdict(list)
        for o in near:
            groups[canon(o.name)].append(o)
        bits = []
        for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(members) == 1:
                bits.append(noun_phrase(evidence, members[0], detail >= 2))
            else:
                bits.append(f"{len(members)} {plural(key, len(members))}")
        where = spatial(near[0], near[0]._depth)
        verb = "are" if (len(near) > 1 or len(groups) > 1) else "is"
        sentences.append(f"{join(bits).capitalize()} {verb} {where}.")

    # ---- 4. background, as a count not a list ------------------------
    if others:
        groups = defaultdict(list)
        for o in others:
            groups[canon(o.name)].append(o)
        bits = [count_phrase(k, len(v)) if len(v) < 4
                else f"several {plural(k, 2)}"
                for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
        sentences.append(f"{join(bits).capitalize()} "
                         f"{'are' if len(others) > 1 else 'is'} "
                         f"visible further back.")

    # ---- 5. anything flagged that has not been said ------------------
    flagged = [o for o in ents[1:] if getattr(o, "checks", None)]
    for o in flagged[:1]:
        hits = ", ".join(n.replace("_", " ") for n in o.checks)
        sentences.append(f"The {o.name} {spatial(o, o._depth)} shows "
                         f"{hits}.")

    # ---- 6. alerts lead ----------------------------------------------
    alerts = alert_lines(evidence)
    if alerts:
        sentences.insert(0, "ALERT: " + "; ".join(alerts[:3]) + ".")

    return " ".join(sentences)


# --------------------------------------------------------------------------- #
def compose_report(evidence, **kw) -> str:
    """Terse incident-report register, for logs rather than prose."""
    skip = attached_ids(evidence)
    lines = []
    sc = (evidence.scene or {}).get("clip_names") or []
    if sc:
        lines.append(f"SCENE: {sc[0][0]}")
    for o in evidence.objects:
        if category(o.name) in NEVER_CAPTION or o.idx in skip:
            continue
        if not o.clip_names:
            continue
        o._frame_h = evidence.height
        parts = [noun_phrase(evidence, o, True, article=False)]
        t = trailing_clause(evidence, o)
        if t:
            parts.append(t)
        parts.append(spatial(o, prominence(o)))
        parts.append(f"conf {o.clip_names[0][1]:.2f}")
        lines.append("- " + " | ".join(parts))
    for a in (evidence.anomalies or []):
        if a.get("triggered"):
            lines.append(f"! {a['name']} {a['prob']:.0%} [{a['where']}]")
    return "\n".join(lines)
