"""Categories, merging and importance — what turns boxes into evidence.

CATEGORIES decide what a detection *is* to the caption. The important split
is entities vs parts: a face is not a thing in the scene, it is part of a
person. Garments and accessories are spoken as "wearing X", never listed.

MERGING collapses boxes that refer to the same physical object. Geometry
decides first — two boxes sitting on top of each other are the same object
whether or not the labels agree.

IMPORTANCE is an explicit policy, not a learned score. Every term is one
line, so when the ranking is wrong you can see which number to change.
"""
from __future__ import annotations
import math

from . import threat

# --------------------------------------------------------------------------- #
#  canonical names
# --------------------------------------------------------------------------- #
SYNONYM = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "baby": "person", "teenager": "person",
    "pedestrian": "person", "worker": "person", "labourer": "person",
    "commuter": "person", "shopper": "person", "customer": "person",
    "student": "person", "factory worker": "person", "human": "person",
    "construction worker": "person", "driver": "person", "rider": "person",
    "elderly man": "person", "elderly woman": "person",
    "security guard": "person", "watchman": "person", "gatekeeper": "person",
    "police officer": "person", "policeman": "person", "policewoman": "person",
    "traffic police officer": "person", "delivery person": "person",
    "delivery rider": "person", "courier": "person", "street vendor": "person",
    "hawker": "person", "engineer": "person", "technician": "person",
    "supervisor": "person", "mechanic": "person", "cleaner": "person",
    "group of people": "person", "crowd of people": "person",
    "crowd": "person", "group of people standing": "person",
    "motorbike": "motorcycle", "two wheeler": "motorcycle",
    "scooty": "scooter", "cycle": "bicycle",
    "autorickshaw": "auto rickshaw", "three wheeler": "auto rickshaw",
    "electric rickshaw": "auto rickshaw", "e rickshaw": "auto rickshaw",
    "lorry": "truck", "mini truck": "truck", "tempo": "truck",
    "container truck": "truck", "tanker truck": "truck",
    "hatchback car": "car", "sedan car": "car", "suv": "car", "taxi": "car",
    "machinery": "machine", "industrial machine": "machine",
    "vehicle number plate": "number plate", "registration plate": "number plate",
    "large knife": "knife", "machete": "knife", "sword": "knife",
    "sickle": "knife", "pistol": "gun", "rifle": "gun",
    "iron rod": "stick", "metal pipe": "stick", "wooden stick": "stick",
    "baseball bat": "stick", "hockey stick": "stick", "lathi": "stick",
    "stray dog": "dog", "stray cattle": "cow", "cattle on the road": "cow",
    # every phrasing CLIP/World might produce for a wrecked vehicle collapses
    # to one canonical name — this is what the [accident] check gates on, so
    # a new synonym here (not a new anomalies.txt entry) is the fix if
    # "accident" still doesn't fire on some other wording.
    "damaged car": "damaged vehicle", "wrecked car": "damaged vehicle",
    "crashed car": "damaged vehicle", "totaled car": "damaged vehicle",
    "crumpled car": "damaged vehicle", "wrecked vehicle": "damaged vehicle",
    "crashed vehicle": "damaged vehicle", "accident vehicle": "damaged vehicle",
    "car wreck": "damaged vehicle", "car crash": "damaged vehicle",
    "overturned car": "overturned vehicle", "flipped car": "overturned vehicle",
    "rolled over car": "overturned vehicle", "capsized vehicle": "overturned vehicle",
}

# --------------------------------------------------------------------------- #
#  category membership
# --------------------------------------------------------------------------- #
# Body parts are EVIDENCE, never entities. A detected face tells you a person
# is present and facing the camera; it is not a thing in the scene.
BODY_PART = {"face", "covered face", "head", "hair", "hand", "arm", "leg",
             "foot", "eye", "mouth", "nose", "ear", "torso", "shoulder"}

# Worn things are spoken as "a woman wearing X", never listed on their own.
GARMENT = {"shirt", "t-shirt", "blouse", "kurta", "kurti", "saree",
           "salwar kameez", "dupatta", "lehenga", "sherwani", "jacket",
           "coat", "raincoat", "hoodie", "sweater", "shawl", "vest",
           "safety vest", "high visibility jacket", "reflective jacket",
           "uniform", "security uniform", "school uniform", "police uniform",
           "delivery uniform", "overalls", "coveralls", "apron", "trousers",
           "jeans", "shorts", "skirt", "dress", "lungi", "dhoti", "gown",
           "graduation gown", "stole", "sash", "scarf", "shoe", "sandal",
           "slipper", "chappal", "boot", "safety boots", "sneaker", "sock"}

ACCESSORY = {"chain", "necklace", "bangle", "ring", "earring", "bracelet",
             "watch", "glasses", "sunglasses", "goggles", "belt", "tie",
             "lanyard", "id card", "badge", "cap", "hat", "topi", "turban",
             "pagri", "helmet", "motorcycle helmet", "hard hat",
             "safety helmet", "face mask", "headscarf", "glove"}

# Vehicle parts are to a car what a face is to a person — evidence that the
# car is there and how it sits, never a thing in the scene on their own.
# Without this the caption opens with "Two window grills" because three
# separate boxes on one car outnumbered the car itself.
# NOTE: "window grill" is deliberately NOT here — in Indian housing that is
# a building security feature (intrusion evidence), and it stays in ACCESS.
# It misfired on a car window here only because confidence was 0.23/0.33;
# that is a threshold problem, not a category one.
VEHICLE_PART = {"car door", "open car boot", "vehicle door", "door handle",
                "windscreen",
                "windshield", "car window", "bonnet", "hood",
                "boot", "trunk", "bumper", "headlight", "tail light",
                "taillight", "wheel", "tyre", "tire", "side mirror",
                "rear view mirror", "wiper", "exhaust", "car seat",
                "steering wheel", "roof rack", "car bonnet"}

PEOPLE = {"person"}
WEAPONS = {"knife", "gun", "weapon", "stick", "baton", "axe",
           "brick", "stone", "glass bottle", "petrol can"}
VEHICLES = {"car", "truck", "bus", "van", "motorcycle", "scooter", "bicycle",
            "auto rickshaw", "cycle rickshaw", "tractor", "ambulance", "jeep",
            "bullock cart", "handcart", "push cart", "school bus",
            "police vehicle", "fire engine", "water tanker", "forklift",
            "damaged vehicle", "overturned vehicle", "train"}
CARRIED = {"bag", "backpack", "handbag", "suitcase", "trolley bag",
           "shoulder bag", "sling bag", "school bag", "box", "carton",
           "cardboard box", "parcel", "sack", "gas cylinder", "cylinder",
           "mobile phone", "laptop", "umbrella", "bottle", "water bottle",
           "shopping bag", "plastic bag", "jewellery", "cash",
           "unattended bag on the ground", "food delivery bag"}
ANIMALS = {"dog", "cat", "cow", "buffalo", "goat", "sheep", "pig", "horse",
           "donkey", "monkey", "bird", "crow", "pigeon", "chicken", "rat"}
ACCESS = {"gate", "main gate", "society gate", "entrance gate", "exit gate",
          "boom barrier", "sliding gate", "turnstile", "metal detector",
          "baggage scanner", "security cabin", "guard booth", "guard post",
          "intercom panel", "biometric scanner",
          "door", "glass door", "shutter", "window", "window grill",
          "compound wall", "fence", "barbed wire", "number plate",
          "cctv camera", "security camera", "atm machine", "billing counter",
          "cash counter", "reception desk", "ladder", "staircase"}
SCENERY = {"sky", "cloud", "sun", "moon", "road", "street", "highway",
           "wall", "floor", "ceiling", "grass", "lawn", "tree", "bush",
           "plant", "leaf", "building", "apartment building", "pavement",
           "footpath", "soil", "mud", "sand", "tiled floor", "field",
           "mountain", "hill", "river", "sea", "beach", "snow", "ice",
           "parked vehicles", "parking area"}

CATEGORY_WEIGHT = {
    "weapon": 2.00,
    "person": 1.00,
    "vehicle": 0.75,
    "carried": 0.70,
    "animal": 0.55,
    "access": 0.45,
    "other": 0.40,
    "garment": 0.30,     # only ever spoken via a wearing relation
    "accessory": 0.30,
    "scenery": 0.05,
    "part": 0.00,        # never captioned
    "vehicle_part": 0.00,
}

W_AREA, W_CENTER, W_OBJECTNESS, W_NAME = 0.25, 0.20, 0.30, 0.25
ANOMALY_BONUS = 0.60

NEVER_CAPTION = ("part", "vehicle_part", "scenery")


def canon(name: str) -> str:
    n = (name or "").lower().strip()
    return SYNONYM.get(n, n)


def threat_name(name: str) -> str:
    """The name threat.py should see — RAW, not merge-canonical.

    canon() folds machete/sword/sickle into "knife" so the caption can say
    "two knives" instead of "a machete and a sword". Feeding that same
    collapsed name into threat.tier_of() would drop a machete from Tier 1
    (purpose-built, alert on sight) into Tier 2 (dual-use, needs an
    aggressive person first) — so threat lookups use the raw name, falling
    back to canon() only for names threat.py doesn't recognise directly.
    """
    n = (name or "").lower().strip()
    return n if threat.tier_of(n) is not None else canon(name)


def category(name: str) -> str:
    c = canon(name)
    if c in BODY_PART:
        return "part"
    if c in VEHICLE_PART:
        return "vehicle_part"
    if c in GARMENT:
        return "garment"
    if c in ACCESSORY:
        return "accessory"
    if c in WEAPONS or threat.tier_of(threat_name(name)) is not None:
        return "weapon"
    if c in PEOPLE:
        return "person"
    if c in VEHICLES:
        return "vehicle"
    if c in CARRIED:
        return "carried"
    if c in ANIMALS:
        return "animal"
    if c in ACCESS:
        return "access"
    if c in SCENERY:
        return "scenery"
    return "other"


# --------------------------------------------------------------------------- #
#  merging
# --------------------------------------------------------------------------- #
def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - i
    return (i / u if u > 0 else 0.0), i, aa, bb


def merge_duplicates(objs, iou_thresh=0.30, contain_thresh=0.70,
                     same_object_iou=0.55):
    """Collapse boxes that refer to the same physical object.

      1. IoU above same_object_iou -> the SAME object, whatever the labels
         say. YOLO's "person", World's "woman" and CLIP's "girl" on one body
         are one entity. The strongest detection keeps its name.
      2. Weaker overlap merges only when the canonical names agree.
      3. Containment merges only when the names agree — so a knife inside a
         person box is never absorbed into the person.
    """
    def strength(o):
        s = o.clip_names[0][1] if o.clip_names else 0.0
        return s * max(o.objectness, 1e-3)

    order = sorted(objs, key=strength, reverse=True)
    kept, counts = [], {}

    for o in order:
        c = canon(o.name)
        hit = None
        for k in kept:
            iou, inter, ao, ak = _iou(o.box, k.box)
            same_name = canon(k.name) == c
            if iou >= same_object_iou:
                hit = k
                break
            if same_name and (iou > iou_thresh or
                              inter / max(min(ao, ak), 1e-9) > contain_thresh):
                hit = k
                break
        if hit is None:
            kept.append(o)
            counts[id(o)] = 1
        else:
            counts[id(hit)] = counts.get(id(hit), 1) + 1
            hit.box = (min(hit.box[0], o.box[0]), min(hit.box[1], o.box[1]),
                       max(hit.box[2], o.box[2]), max(hit.box[3], o.box[3]))
            hit.objectness = max(hit.objectness, o.objectness)
            if o.clip_names and o.clip_names[0][0] != hit.name:
                hit.clip_names = (hit.clip_names[:1] + o.clip_names[:1]
                                  + hit.clip_names[1:3])

    for k in kept:
        k.merged_count = counts.get(id(k), 1)
    return kept


# --------------------------------------------------------------------------- #
#  importance
# --------------------------------------------------------------------------- #
def importance(o, W, H, anomaly_tiles=None) -> float:
    area = min(1.0, math.sqrt(max(o.area_frac, 0.0)) * 2.2)

    x0, y0, x1, y1 = o.box
    cx, cy = ((x0 + x1) / 2) / max(W, 1), ((y0 + y1) / 2) / max(H, 1)
    center = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.5) / 0.707)

    obj = min(1.0, max(0.0, o.objectness))
    name_s = o.clip_names[0][1] if o.clip_names else 0.0
    name = min(1.0, name_s * 2.0 + o.margin)

    base = (W_AREA * area + W_CENTER * center
            + W_OBJECTNESS * obj + W_NAME * name)
    t = getattr(o, "threat", None)
    if t and t.get("weight") is not None:
        score = t["weight"] * base           # tiered, idle vs held
    else:
        score = CATEGORY_WEIGHT.get(category(o.name), 0.4) * base

    if anomaly_tiles:
        for (tx0, ty0, tx1, ty1) in anomaly_tiles:
            ix = max(0.0, min(x1, tx1) - max(x0, tx0))
            iy = max(0.0, min(y1, ty1) - max(y0, ty0))
            if ix * iy > 0.25 * max((x1 - x0) * (y1 - y0), 1e-9):
                score += ANOMALY_BONUS
                break
    return round(score, 4)


def _held_by(o, objs, min_ratio=0.5):
    """Cheap containment check: is o mostly inside a person's box?

    The full holding relation is derived later (relations.py runs after
    rank(), since it needs the renumbered idx). Threat assessment needs to
    know idle-vs-held right now, so it gets its own quick geometric check
    rather than waiting.
    """
    ox0, oy0, ox1, oy1 = o.box
    oa = max(1e-9, (ox1 - ox0) * (oy1 - oy0))
    for p in objs:
        if p is o or category(p.name) != "person":
            continue
        px0, py0, px1, py1 = p.box
        ix0, iy0 = max(ox0, px0), max(oy0, py0)
        ix1, iy1 = min(ox1, px1), min(oy1, py1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter / oa >= min_ratio:
            return p
    return None


def rank(objs, W, H, anomaly_tiles=None):
    for o in objs:
        tname = threat_name(o.name)
        if threat.tier_of(tname) is not None:
            holder = _held_by(o, objs)
            flags = set((holder.checks or {}).keys()) if holder else set()
            conf = o.clip_names[0][1] if o.clip_names else 1.0
            o.threat = threat.assess(tname, held_by_person=holder is not None,
                                     person_flags=flags, confidence=conf)
        else:
            o.threat = None

    for o in objs:
        o.importance = importance(o, W, H, anomaly_tiles)
    objs.sort(key=lambda o: -o.importance)
    for i, o in enumerate(objs):
        o.idx = i + 1
    return objs


def explain(o, W, H) -> dict:
    area = min(1.0, math.sqrt(max(o.area_frac, 0.0)) * 2.2)
    x0, y0, x1, y1 = o.box
    cx, cy = ((x0 + x1) / 2) / max(W, 1), ((y0 + y1) / 2) / max(H, 1)
    center = 1.0 - min(1.0, math.hypot(cx - 0.5, cy - 0.5) / 0.707)
    name_s = o.clip_names[0][1] if o.clip_names else 0.0
    cat = category(o.name)
    return {"category": cat,
            "cat_weight": CATEGORY_WEIGHT.get(cat, 0.4),
            "area": round(area, 3),
            "center": round(center, 3),
            "objectness": round(o.objectness, 3),
            "name_conf": round(min(1.0, name_s * 2 + o.margin), 3),
            "total": getattr(o, "importance", None)}
