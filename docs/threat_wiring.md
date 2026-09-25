# Wiring the threat taxonomy in

`threat.py` goes in `jana2/`. Three edits.

---

## 1. `jana2/postprocess.py` — replace the WEAPONS block

```python
from .threat import ALL_THREAT, tier_of, IDLE_WEIGHT

# WEAPONS is gone. Threat objects are categorised by tier, and their weight
# depends on whether anyone is holding them — see threat.assess().
```

In `category()`:

```python
def category(name: str) -> str:
    c = canon(name)
    if c in BODY_PART:
        return "part"
    if c in GARMENT:
        return "garment"
    if c in ACCESSORY:
        return "accessory"
    t = tier_of(c)
    if t:
        return f"threat:{t}"       # threat:firearm, threat:improvised, ...
    if c in PEOPLE:
        return "person"
    ...
```

and in `CATEGORY_WEIGHT`, replace `"weapon": 2.0` with the idle weights:

```python
    "threat:firearm": 3.00,
    "threat:blade": 2.20,
    "threat:incendiary": 1.30,
    "threat:dual_use": 0.55,
    "threat:improvised": 0.40,
```

Those are the **idle** weights — the object lying there, untouched. That
single change is what stops a yellow guard rail outranking a worker.

---

## 2. `jana2/naming.py` — promote held objects

After relations are derived (they tell us what is in a hand), re-weight and
attach the assessment. Insert right after the `derive(...)` call:

```python
    # a held object is a different object. Re-assess and re-rank.
    from .threat import assess
    held_by = {}
    for r in rels:
        if r["predicate"] == "holding":
            held_by[r["object"]] = r["subject"]

    changed = False
    for o in objs:
        flags = set()
        holder = held_by.get(o.idx)
        if holder:
            for h in objs:
                if h.idx == holder:
                    flags = set((h.checks or {}).keys())
                    break
        a = assess(canon(o.name), holder is not None, flags,
                   o.clip_names[0][1] if o.clip_names else 0.0)
        if a:
            o.threat = a
            changed = True

    if changed:
        objs = rank(objs, W, H, hot)
        rels = derive(objs, W, H, CFG.max_relations)   # idx changed
```

Add `threat: dict = field(default_factory=dict)` to `ObjectSlot` in
`evidence.py`.

---

## 3. `jana2/readout.py` — alerts by level, not by mere presence

In `alert_lines()`, add before the per-object check loop:

```python
    for o in evidence.objects:
        t = getattr(o, "threat", None)
        if t and t.get("level") in ("confirmed", "probable"):
            out.append((SEV_RANK.get(t["severity"], 1), -1.0,
                        f"{t['level']} {t['tier']}: {o.name} "
                        f"[#{o.idx}] — {t['reason']}"))
```

Only `confirmed` and `probable` raise an alert. `possible` and `context`
appear in the evidence table but do not interrupt anyone — that distinction
is the whole point of the tiering.

---

## 4. `assets/world_classes.txt` — the detector needs the words

A taxonomy is useless if nothing proposes a box for it. Replace the weapons
section with:

```
knife
large knife
machete
chopper
sword
sickle
axe
dagger
cleaver
gun
pistol
revolver
rifle
country made pistol
wooden stick
lathi
iron rod
metal pipe
steel rod
cricket bat
hockey stick
baseball bat
hammer
crowbar
brick
stone in hand
glass bottle in hand
chain
petrol can
person holding a knife
person holding a stick
person raising a weapon
```

Those last three are phrase classes — YOLO-World can detect the *situation*
directly, which is often easier than detecting a 30-pixel blade.

---

## What changes in practice

Before, on `factory.jpg`:

    A wooden stick is on the right. A middle aged man wearing helmet…

After: the rail is `threat:improvised`, nobody is holding it, weight 0.40 —
so the worker leads and the rail is barely mentioned. No `holding baton`
relation surfacing as an alert.

And on the machete frame: `blade` tier, held, aggravating flags from the
person check → `confirmed blade` with a high-severity alert.

---

## One judgement call worth revisiting

`sickle` and `axe` are in DUAL_USE, so an axe sitting on a worksite is not an
alert. In a rural or agricultural deployment that is correct. If you deploy
somewhere an axe is never legitimate, move them into `BLADE_WEAPON` — it is a
one-line change and it is a deployment decision, not a modelling one.

Same for `lathi`: every guard in the country carries one, so it is improvised
and idle-weighted. If a site has no guards, promote it.
