"""PATCH for jana2/proposals.py — replace suppress_parts() and its call site.

Why: a knife held by a person is 100% inside the person's box. The old rule
("drop anything mostly inside a bigger box") deleted it by construction. That
is the difference between seeing a weapon and not.

New rule: drop the child only if it is BOTH mostly inside the parent AND
close to the parent's size — i.e. a near-duplicate box, not a small distinct
object. A wheel inside a car is ~50%+ of the car box; a knife is ~5% of a
person box.
"""

# ---- REPLACE the whole suppress_parts function with this -------------------
def suppress_parts(boxes, thresh, frame_area, max_parent_frac=1.0,
                   min_child_ratio=0.55, protected=None):
    """Drop near-duplicate nested boxes; keep small distinct objects.

    thresh          : fraction of the child that must lie inside the parent
    max_parent_frac : a box bigger than this share of the frame is scenery
                      and absorbs nothing
    min_child_ratio : child_area / parent_area must exceed this to count as a
                      duplicate. Small held objects fall below it and survive.
    """
    protected = protected or set()
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
    keep = []
    for i in order:
        ai = _area(boxes[i])
        if ai <= 0:
            continue
        if i in protected:
            keep.append(i)
            continue
        swallowed = False
        for j in keep:
            aj = _area(boxes[j])
            if aj / frame_area > max_parent_frac:
                continue                       # scenery, not a parent
            if ai / max(aj, 1e-9) < min_child_ratio:
                continue                       # small distinct object — keep
            if _inter(boxes[i], boxes[j]) / ai >= thresh:
                swallowed = True
                break
        if not swallowed:
            keep.append(i)
    return keep


# ---- REPLACE the containment block inside Proposer.__call__ ----------------
#
#         kb = [boxes[i] for i in keep]
#         prot = {n for n, i in enumerate(keep)
#                 if self.cfg.protect_yolo and srcs[i] == "yolo"}
#         k = suppress_parts(kb, self.cfg.containment_thresh, frame,
#                            self.cfg.containment_max_parent,
#                            self.cfg.containment_min_child_ratio, prot)
#         keep = [keep[i] for i in k]
#         st.after_containment = len(keep)
