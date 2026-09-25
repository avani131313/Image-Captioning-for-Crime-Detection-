# Wiring the composer in

`compose.py` goes in `jana2/`. It imports from `readout.py`, so keep both.

## `app.py` — replace the caption block

```python
from jana2 import compose

with right:
    st.subheader("operator view")
    lines = readout.operator_lines(ev, top_n, min_score, min_obj)
    st.code("\n".join(lines) if lines else "nothing above thresholds",
            language=None)

    st.subheader("caption")
    st.info(compose.compose(ev, min_score, min_obj, top_n))

    with st.expander("incident report format"):
        st.code(compose.compose_report(ev), language=None)

    with st.expander("old flat caption (for comparison)"):
        st.write(readout.caption(ev, min_score, min_obj, top_n))
```

Keeping the old one visible is worth it for a while — when the composer drops
something, the flat list shows you what it dropped.

## `scripts/probe.py` and `scripts/snapshot.py`

Swap `readout.caption(ev)` for `compose.compose(ev)` wherever it appears.

## No rebuild needed

Pure code. Press **R** in the browser; **C** first only if you also changed
an asset.

---

# What it does differently

**Depth.** Small and high in the frame means far away, because cameras look
outward and down. So a wide society shot becomes *"several people are visible
further back"* instead of *"eight people"*. This alone fixes the worst thing
about the society courtyard caption.

**One subject.** The highest-importance entity leads and gets the full
treatment — attributes, what it's wearing, what it's holding, what it's on.
Everything else is context, grouped and counted.

**Verbs from evidence.** `pose=walking` gives "is walking"; a `riding`
relation gives "is riding"; a vehicle gets "is parked". Sentences stop all
beginning "There is".

**Grouping by depth, not just by name.** Foreground entities get described;
background ones get counted.

Rough shape of the output:

    A residential building compound. A young man in a dark shirt stands in
    the centre, holding a long blade. Two cars are on the left. Several
    people are visible further back.

---

# Two honest limits

**`prominence()` is a heuristic, not depth.** Small-and-high usually means
far, but a CCTV camera mounted high and looking down breaks that — distant
people appear in the *middle* of the frame, not the top. If your cameras are
mounted that way, the rule needs the frame's horizon line, which we don't
have. Watch whether "further back" is ever wrong on your footage; if it is,
drop the depth grouping rather than trusting it.

**The subject is whoever `importance` ranks first.** If that ranking is
wrong, the caption leads with the wrong thing — and a caption that opens on
the wrong subject reads worse than a flat list. The "why this ranking?" panel
is where to check it, and `CATEGORY_WEIGHT` in `postprocess.py` is where to
change it.

---

# On the copy-constrained decoder

Run this first. My honest expectation is that it gets you most of the way,
and that the remaining gap is smaller than the cost of training and
maintaining a decoder without an eval set.

If after looking at 20 images the complaint is "the facts are right but it
reads mechanically", that is when a decoder earns its place — and by then
you will have the evidence-to-caption pairs to train it on, because every
image you run through this produces one.
