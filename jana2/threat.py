"""Weapon taxonomy and threat assessment.

The mistake in a flat weapon list is treating a revolver and a brick as the
same thing. Almost every object that can be a weapon is also something else —
a sickle is a farm tool, a hammer is a hammer, a lathi is what every guard in
the country carries. What separates a tool from a weapon is usually not the
object, it is who is holding it and how.

So objects are tiered by how much context they need before they mean
anything, and the threat level is a function of the object AND the situation:

    TIER 1  purpose-built. Presence alone is the alert.
            firearms, machete, sword, dagger.
    TIER 2  dual-use. Extremely common as tools. Only a weapon when held by
            a person, and serious only when the person also reads as
            aggressive or concealed.
            knife, axe, sickle, hammer, crowbar, scissors.
    TIER 3  improvised. Ordinary objects. A stick lying on the ground is a
            stick. In a raised hand during a crowd event it is a weapon.
            lathi, iron rod, rebar, cricket bat, brick, bottle, chain.

This keeps a yellow guard rail from being ranked above a worker, which is
exactly what a flat list did.

Nothing here decides anything on its own — it produces a level and a weight
that the importance ranking and the alert logic consume.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
#  TIER 1 — purpose-built. No legitimate everyday use.
# --------------------------------------------------------------------------- #
FIREARM = {
    "gun", "pistol", "revolver", "rifle", "shotgun", "firearm",
    "handgun", "country made pistol", "katta", "air gun", "pellet gun",
    "carbine", "assault rifle",
}

BLADE_WEAPON = {
    "machete", "chopper", "sword", "talwar", "dagger", "katar",
    "khukri", "kukri", "kirpan", "switchblade", "flick knife",
    "large knife", "butcher knife", "cleaver",
}

# --------------------------------------------------------------------------- #
#  TIER 2 — dual-use. Tool by default, weapon in context.
#  In India especially: sickles and axes are agricultural, lathis are issued
#  to guards, hammers and wrenches are on every worksite.
# --------------------------------------------------------------------------- #
DUAL_USE = {
    "knife", "kitchen knife", "pocket knife", "blade", "razor", "box cutter",
    "axe", "hatchet", "sickle", "hasiya", "scythe",
    "hammer", "sledgehammer", "wrench", "spanner", "crowbar", "pickaxe",
    "screwdriver", "chisel", "scissors", "shears", "saw",
    "syringe", "needle",
}

# --------------------------------------------------------------------------- #
#  TIER 3 — improvised. Meaningless without a person and an action.
# --------------------------------------------------------------------------- #
IMPROVISED = {
    "stick", "wooden stick", "lathi", "danda", "baton", "cane",
    "iron rod", "metal rod", "metal pipe", "rebar", "sariya", "steel bar",
    "cricket bat", "baseball bat", "hockey stick", "golf club",
    "brick", "stone", "rock",
    # "chain" alone is the jewellery/accessory sense (necklace) — kept out of
    # this set on purpose. "metal chain" is the distinct detector class for
    # the weapon sense, so the two never collide.
    "metal chain", "belt", "rope",
    "bottle", "glass bottle", "broken bottle", "beer bottle",
    "helmet in hand", "plank", "chair leg",
}

# chemical / incendiary — rare, but the consequence is severe enough that
# a low-confidence hit still warrants a look
INCENDIARY = {
    "petrol can", "jerry can", "molotov cocktail", "petrol bomb",
    "acid bottle", "kerosene can", "gas cylinder in hand", "lighter fluid",
}

ALL_THREAT = FIREARM | BLADE_WEAPON | DUAL_USE | IMPROVISED | INCENDIARY

# --------------------------------------------------------------------------- #
#  tier metadata
# --------------------------------------------------------------------------- #
TIER = {
    "firearm": FIREARM,
    "blade": BLADE_WEAPON,
    "incendiary": INCENDIARY,
    "dual_use": DUAL_USE,
    "improvised": IMPROVISED,
}

# importance weight when the object is NOT in anyone's hand
IDLE_WEIGHT = {
    "firearm": 3.00,      # a gun on a table is still a gun
    "blade": 2.20,
    "incendiary": 1.30,
    "dual_use": 0.55,     # a hammer on a bench is a hammer
    "improvised": 0.40,   # a rail is a rail
}

# importance weight when a person is holding it
HELD_WEIGHT = {
    "firearm": 3.50,
    "blade": 3.00,
    "incendiary": 2.60,
    "dual_use": 2.00,     # a held knife is a different object entirely
    "improvised": 1.60,
}

# alert severity, given the level below
SEVERITY = {"confirmed": "high", "probable": "high",
            "possible": "medium", "context": "low"}

# person-level checks that escalate a dual-use or improvised object
AGGRAVATING = {"aggressive", "face_hidden", "climbing_person", "weapon_held"}


# --------------------------------------------------------------------------- #
def tier_of(name: str) -> str | None:
    n = (name or "").lower().strip()
    for tier, members in TIER.items():
        if n in members:
            return tier
    return None


def assess(name: str, held_by_person: bool = False,
           person_flags: set | None = None, confidence: float = 1.0) -> dict:
    """What this object means, given the situation.

    Returns {tier, level, severity, weight, reason} or an empty dict when the
    object is not a threat at all.

      confirmed  purpose-built weapon, or a dual-use object held by someone
                 who also reads as aggressive or concealed
      probable   purpose-built weapon at lower confidence, or a held blade
      possible   dual-use object in a hand
      context    improvised object in a hand, nothing else unusual
    """
    tier = tier_of(name)
    if tier is None:
        return {}

    flags = set(person_flags or ())
    aggravated = bool(flags & AGGRAVATING)

    if tier in ("firearm", "blade", "incendiary"):
        level = "confirmed" if confidence >= 0.45 else "probable"
        reason = "purpose-built weapon"
    elif tier == "dual_use":
        if not held_by_person:
            return {"tier": tier, "level": None, "severity": None,
                    "weight": IDLE_WEIGHT[tier],
                    "reason": "tool, not in anyone's hand"}
        level = "confirmed" if aggravated else "possible"
        reason = ("held, and the person reads as aggressive or concealed"
                  if aggravated else "dual-use object in a hand")
    else:                                        # improvised
        if not held_by_person:
            return {"tier": tier, "level": None, "severity": None,
                    "weight": IDLE_WEIGHT[tier],
                    "reason": "ordinary object, not in anyone's hand"}
        level = "probable" if aggravated else "context"
        reason = ("improvised object raised by an aggressive person"
                  if aggravated else "ordinary object in a hand")

    return {
        "tier": tier,
        "level": level,
        "severity": SEVERITY.get(level),
        "weight": (HELD_WEIGHT if held_by_person else IDLE_WEIGHT)[tier],
        "reason": reason,
    }


def phrase(name: str, a: dict) -> str:
    """How to word it in a caption. Never state more than the tier supports."""
    if not a or not a.get("level"):
        return name
    if a["tier"] == "firearm":
        return f"what appears to be a {name}"
    if a["level"] == "confirmed":
        return name
    if a["level"] == "possible":
        return f"a {name} in hand"
    return f"a {name}"
