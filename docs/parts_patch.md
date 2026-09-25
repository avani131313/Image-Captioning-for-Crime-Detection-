# Parts vs entities — stop listing a face next to a person

Three edits. The principle: **a caption is about entities. Detections include
parts. Parts get attached to their owner, never listed beside it.**

---

## 1. `jana2/postprocess.py` — new categories

Add these sets beside the existing ones:

```python
# Body parts are EVIDENCE, never entities. A detected face tells you a person
# is present and facing the camera; it is not a thing in the scene.
BODY_PART = {"face", "covered face", "head", "hair", "hand", "arm", "leg",
             "foot", "eye", "mouth", "nose", "ear", "torso", "shoulder"}

# Worn things are spoken as "a woman wearing X", never as a separate object.
GARMENT = {"shirt", "t-shirt", "blouse", "kurta", "kurti", "saree",
           "salwar kameez", "dupatta", "lehenga", "sherwani", "jacket",
           "coat", "raincoat", "hoodie", "sweater", "shawl", "vest",
           "safety vest", "high visibility jacket", "reflective jacket",
           "uniform", "security uniform", "school uniform", "police uniform",
           "delivery uniform", "overalls", "apron", "trousers", "jeans",
           "shorts", "skirt", "dress", "lungi", "dhoti", "gown",
           "graduation gown", "stole", "sash", "scarf", "shoe", "sandal",
           "slipper", "chappal", "boot", "safety boots", "sneaker", "sock"}

ACCESSORY = {"chain", "necklace", "bangle", "ring", "earring", "bracelet",
             "watch", "glasses", "sunglasses", "goggles", "belt", "tie",
             "lanyard", "id card", "badge", "cap", "hat", "topi", "turban",
             "pagri", "helmet", "motorcycle helmet", "hard hat",
             "safety helmet", "face mask", "headscarf", "glove"}
```

Then extend `category()` — put these **before** the existing checks:

```python
def category(name: str) -> str:
    c = canon(name)
    if c in BODY_PART:
        return "part"
    if c in GARMENT:
        return "garment"
    if c in ACCESSORY:
        return "accessory"
    if c in WEAPONS:
        return "weapon"
    ...
```

And give them weights in `CATEGORY_WEIGHT`:

```python
    "part": 0.0,          # never captioned
    "garment": 0.30,      # only via a wearing relation
    "accessory": 0.30,
```

---

## 2. `jana2/relations.py` — let clothing produce `wearing`

Replace the `WORN` set with:

```python
from .postprocess import GARMENT, ACCESSORY
WORN = GARMENT | ACCESSORY
```

and relax the position rule — a salwar kameez covers the whole body, not just
the top third. In the worn branch:

```python
            # ---- worn on a person -------------------------------------
            if cb == "person" and na in WORN:
                f = _inside_frac(ab, bb)
                if f > 0.55:
                    out.append(_rel(b, "wearing", a, 0.80, inside=round(f, 2)))
                    continue
```

Add a part-of rule so a face binds to its person:

```python
            # ---- body part of a person --------------------------------
            if cb == "person" and category(a.name) == "part":
                f = _inside_frac(ab, bb)
                if f > 0.6:
                    out.append(_rel(b, "has_part", a, 0.9, inside=round(f, 2)))
                    continue
```

---

## 3. `jana2/readout.py` — attach instead of list

In `select()`, drop parts outright and drop anything already attached to a
person:

```python
def select(evidence, min_score=None, min_objectness=None, max_objects=8):
    min_score = CFG.caption_min_score if min_score is None else min_score
    min_objectness = (CFG.caption_min_objectness if min_objectness is None
                      else min_objectness)

    # things spoken via their owner instead of on their own
    attached = {r["object"] for r in (evidence.relations or [])
                if r["predicate"] in ("wearing", "has_part", "holding")}

    objs = []
    for o in evidence.objects:
        if not o.clip_names or o.clip_names[0][1] < min_score:
            continue
        if o.objectness < min_objectness:
            continue
        if category(o.name) in ("scenery", "part"):
            continue
        if o.idx in attached:
            continue
        objs.append(o)
    objs.sort(key=lambda o: -getattr(o, "importance", 0.0))
    return objs[:max_objects]
```

Add a helper that gathers what an entity is wearing or holding:

```python
def _attached_to(evidence, idx):
    worn, held = [], []
    for r in (evidence.relations or []):
        if r["subject"] != idx:
            continue
        if r["predicate"] == "wearing":
            worn.append(r["object_name"])
        elif r["predicate"] == "holding":
            held.append(r["object_name"])
    return worn, held
```

And use the specific name plus its attachments in the listing. Replace the
listing block in `caption()`:

```python
    groups = defaultdict(list)
    for o in objs:
        groups[canon(o.name)].append(o)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    def label(key, members):
        if len(members) == 1:
            o = members[0]
            a = o.attributes or {}
            bits = []
            if "age" in a:
                bits.append(a["age"]["label"].replace(" person", ""))
            nm = a.get("gender", {}).get("label", o.name)
            worn, held = _attached_to(evidence, o.idx)
            s = " ".join(bits + [nm])
            if worn:
                s += " wearing " + join(worn[:2])
            if held:
                s += " holding " + join(held[:2])
            return s
        names = {m.name for m in members}
        return names.pop() if len(names) == 1 else key

    listing = join([count_phrase(label(n, v), len(v)) for n, v in ordered])
```

---

## Result

Before:

    The image shows a person, a salwar kameez, a chain and a face.

After:

    The image shows a young adult woman wearing a salwar kameez and a chain.

The face still exists in the evidence table with its box and score — it is
just no longer a sentence. That is the distinction: everything detected stays
auditable, but only entities get spoken.

---

## Rebuild

```bash
cd /root/avani/code_files
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
.venv/bin/streamlit run app.py --server.port 8612 --server.address 0.0.0.0
```

No asset rebuild needed — these are all code-side category changes.

## One thing to watch

`helmet` is now an ACCESSORY, so on a motorcycle frame it will be absorbed
into "a rider wearing a helmet" instead of being listed. That is right for
captions but it means the `no_helmet` check's `requires=` gate should stay
keyed on `motorcycle`, not on `helmet`.
