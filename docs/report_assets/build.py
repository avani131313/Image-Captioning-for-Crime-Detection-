from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether, HRFlowable)
from reportlab.lib.enums import TA_LEFT

INK=colors.HexColor("#1a1a1a"); MUT=colors.HexColor("#5b5b5b")
ACC=colors.HexColor("#2563eb"); GOOD=colors.HexColor("#059669")
BAD=colors.HexColor("#dc2626"); WARN=colors.HexColor("#d97706")
LIGHT=colors.HexColor("#f1f5f9"); LINE=colors.HexColor("#d7d7d7")

ss=getSampleStyleSheet()
def S(n,**kw):
    base=dict(name=n,fontName="Helvetica",fontSize=9.5,leading=14,textColor=INK)
    base.update(kw); return ParagraphStyle(**base)
H1=S("H1",fontName="Helvetica-Bold",fontSize=19,leading=23,spaceAfter=4)
H2=S("H2",fontName="Helvetica-Bold",fontSize=13,leading=17,spaceBefore=13,spaceAfter=5,textColor=ACC)
H3=S("H3",fontName="Helvetica-Bold",fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=3)
BODY=S("BODY",spaceAfter=6)
SMALL=S("SMALL",fontSize=8.3,leading=11.5,textColor=MUT)
CAP=S("CAP",fontSize=8,leading=11,textColor=MUT,spaceBefore=2,spaceAfter=10)
CODE=S("CODE",fontName="Courier",fontSize=8.2,leading=11.5,textColor=colors.HexColor("#0f172a"))
LEAD=S("LEAD",fontSize=11,leading=16,textColor=MUT,spaceAfter=10)

def tbl(data,widths,head=True,fs=8.5,align=None):
    t=Table(data,colWidths=widths,repeatRows=1 if head else 0)
    st=[("FONT",(0,0),(-1,-1),"Helvetica",fs),
        ("TEXTCOLOR",(0,0),(-1,-1),INK),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("LINEBELOW",(0,0),(-1,-2),0.4,LINE)]
    if head:
        st+=[("FONT",(0,0),(-1,0),"Helvetica-Bold",fs),
             ("BACKGROUND",(0,0),(-1,0),LIGHT),
             ("LINEBELOW",(0,0),(-1,0),0.8,ACC)]
    if align: st+=align
    t.setStyle(TableStyle(st)); return t

def callout(title,body,col=ACC,bg="#eff6ff"):
    inner=[[Paragraph(f"<b>{title}</b>",S("x",fontName="Helvetica-Bold",fontSize=9.5,textColor=col))],
           [Paragraph(body,S("y",fontSize=9,leading=13))]]
    t=Table(inner,colWidths=[165*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor(bg)),
        ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LINEBEFORE",(0,0),(0,-1),2.5,col)]))
    return t

def img(p,w=165*mm):
    from PIL import Image as PI
    iw,ih=PI.open(p).size
    return Image(p,width=w,height=w*ih/iw)

def footer(canv,doc):
    canv.saveState()
    canv.setFont("Helvetica",7.5); canv.setFillColor(MUT)
    canv.drawString(22*mm,12*mm,"JANA2 — Object-Centric Surveillance Captioning · Internship Report")
    canv.drawRightString(188*mm,12*mm,f"{doc.page}")
    canv.setStrokeColor(LINE); canv.setLineWidth(0.4)
    canv.line(22*mm,15*mm,188*mm,15*mm)
    canv.restoreState()

E=[]
# ============ COVER ============
E.append(Spacer(1,18*mm))
E.append(Paragraph("JANA2", S("t",fontName="Helvetica-Bold",fontSize=40,leading=44,textColor=ACC)))
E.append(Paragraph("Object-Centric Image Captioning for Indian Video Surveillance",
                   S("s",fontName="Helvetica",fontSize=14,leading=19,textColor=INK,spaceAfter=6)))
E.append(Paragraph("Building a crime and incident detection system that explains itself",
                   S("s2",fontSize=10.5,textColor=MUT,spaceAfter=14)))
E.append(HRFlowable(width="100%",thickness=1.6,color=ACC,spaceAfter=12))
E.append(Paragraph(
 "This report documents the design, failures, diagnoses and measured results of a "
 "surveillance captioning system built from scratch. Every claim here is backed by a "
 "number that was measured on this project, not estimated.", LEAD))

E.append(Paragraph("Headline results", H2))
E.append(tbl([
 ["Outcome","Measurement","Method"],
 ["Alert quality transformed","separation 0.03 → 0.86","replaced hand-written phrase pairs with learned linear probes"],
 ["8 of 9 checks now usable","AP 0.80 – 0.95","supervised probes fitted on VLM-verified labels"],
 ["Encoder 7x smaller","93M → 13.4M, ~3% AP loss","knowledge distillation with a task-aware loss"],
 ["Whole system inside budget","109M → 29M","against a 100M ceiling"],
 ["Labels created from zero","45,891 usable frame labels","weak supervision via Moondream3, no human annotation"],
 ["Thresholds measured","2 false alarms / 1000 frames","calibrated on 400 real CCTV frames"],
], [42*mm,40*mm,83*mm], fs=8.3))

E.append(Spacer(1,8))
E.append(callout("The single most important lesson",
 "Three separate bugs this project had the same shape: <b>a relative ranking was treated as "
 "evidence</b>. A softmax always elects a winner, so the model reported a scene, a name and an "
 "attribute with confidence even when nothing in its list matched. The fix in every case was an "
 "<b>absolute floor</b> — a raw similarity below which the system says nothing rather than "
 "guessing. Ranking answers <i>which is most likely</i>; it never answers <i>is any of this true</i>."))
E.append(PageBreak())

# ============ 1 PROBLEM & ARCHITECTURE ============
E.append(Paragraph("1 · The problem and the architecture",H1))
E.append(Paragraph(
 "The system takes a still frame from an Indian surveillance camera — a housing-society gate, a "
 "road, a mall entrance, a factory floor — and produces two things: a caption a person can read, "
 "and a list of alerts an operator can act on. It must be able to say <i>why</i> it said what it "
 "said.",BODY))
E.append(Paragraph("Why there is no language model in the pipeline",H3))
E.append(Paragraph(
 "The predecessor project used a caption decoder and failed for a specific reason: the decoder "
 "overrode correct evidence. The perception stage identified a cat; the decoder wrote \"dog\", "
 "because that sentence was more probable. Fluency was optimised, truth was not.",BODY))
E.append(Paragraph(
 "JANA2 removes the decoder entirely. Every word in the output traces to a rule that can be read "
 "in source. The cost is prose that is merely serviceable. The benefit is that a wrong word can "
 "always be traced to a specific line — which is the property a security system actually needs.",BODY))
E.append(img("pipeline.png"))
E.append(Paragraph("Figure 1 — Detectors supply boxes and coarse labels; CLIP supplies every judgement; "
                   "a deterministic composer assembles the output.",CAP))

E.append(Paragraph("The three readouts",H3))
E.append(Paragraph(
 "One early insight shaped everything after: <b>different questions need different mathematics</b>, "
 "and using the wrong one produces confident nonsense rather than an error.",BODY))
E.append(tbl([
 ["Question","Readout","Why"],
 ["What is this object?","softmax over 725 names","many mutually exclusive options compete"],
 ["What colour is the shirt?","softmax within a group","only ~15 options are relevant"],
 ["Is there a fire?","contrastive pair, then probe","a yes/no claim against its negation"],
],[45*mm,45*mm,75*mm],fs=8.5))
E.append(Spacer(1,6))
E.append(callout("What went wrong before this was understood",
 "Naming through a sigmoid gave near-zero scores for correct answers. Yes/no checks scored through "
 "SigLIP's native logit scale (~100) turned a cosine gap of 0.05 into 99% confidence — a studio "
 "portrait produced <i>no helmet 99%</i>, <i>crowd surge 87%</i> and <i>darkness 92%</i> "
 "simultaneously.",WARN,"#fffbeb"))
E.append(PageBreak())

# ============ 2 TIMELINE ============
E.append(Paragraph("2 · How the work progressed",H1))
E.append(img("timeline.png"))
E.append(Paragraph("Figure 2 — Roughly half the effort went to diagnosing failures rather than "
                   "adding features, which is normal and worth expecting.",CAP))
E.append(Paragraph("Phases in detail",H3))
E.append(tbl([
 ["Phase","What happened","What it produced"],
 ["Environment","CUDA 12.4 driver vs newer builds; torch reinstalled repeatedly",
  "A hard constraint that shaped every later tool choice"],
 ["Architecture","Dropped the class-agnostic RPN; settled on yolo26n + YOLO-World-s",
  "Fewer nameless boxes, fewer bad zero-shot guesses"],
 ["Bug hunt","Duplicate detections, body parts as entities, flat weapon list",
  "Geometry-first merging, part/entity split, tiered threat model"],
 ["Readout fixes","softmax vs sigmoid vs contrastive pairs; gating; temperature",
  "False alarms on portraits eliminated"],
 ["Distillation","154k crops harvested, projector trained on frozen backbone",
  "93M encoder replaced by 13.4M"],
 ["Weak supervision","96GB UCF-Crime, 56k frames, Moondream3 via vLLM",
  "45,891 frame-level labels from zero human annotation"],
 ["Probes","Logistic probes fitted per check, gated on AP",
  "8 checks moved from guesswork to measurement"],
 ["Calibration","400 real CCTV frames, thresholds at a false-alarm budget",
  "Thresholds derived from data instead of intuition"],
],[26*mm,68*mm,71*mm],fs=8))
E.append(PageBreak())

# ============ 3 BUGS ============
E.append(Paragraph("3 · Failures, diagnoses and fixes",H1))
E.append(Paragraph("This is the most instructive section. Each of these cost real time, and the "
                   "diagnosis mattered more than the fix.",LEAD))

def bug(n,title,symptom,cause,fix,lesson):
    E.append(Paragraph(f"{n} · {title}",H3))
    E.append(tbl([
      ["Symptom",symptom],["Root cause",cause],["Fix",fix],
    ],[24*mm,141*mm],head=False,fs=8.5))
    E.append(Paragraph(f"<b>Lesson.</b> {lesson}",SMALL))
    E.append(Spacer(1,7))

bug("3.1","Every check reported 61%",
 "Unrelated checks all fired at ~61% on ordinary images.",
 "prob = sigmoid(margin × T) returns <b>0.5 when there is no evidence at all</b>. A threshold of "
 "0.60 therefore fired on a cosine margin of 0.022 — indistinguishable from noise. The "
 "'absolute floor' meant to prevent this was set at 0.02, the same value, so it never bound.",
 "Rescaled so no-evidence reads 0.0, and retuned temperature to match.",
 "Know where your scale starts. A probability that is 0.5 for 'nothing here' will fire on nothing.")

bug("3.2","A parked motorcycle was 'a road accident scene'",
 "Scene classification asserted an accident on an empty driveway.",
 "Scene is a softmax over 40 labels: it <b>always</b> elects a winner. With no matching place in "
 "the list, the nearest one won and was stated as fact.",
 "Added a raw-cosine floor; below it the caption says nothing about location. Also removed the "
 "event label from a vocabulary of places.",
 "A forced choice among wrong options still returns an answer. Absolute evidence, not rank.")

bug("3.3","A dustbin became a petrol can — and then a weapon",
 "Ordinary bins raised incendiary threat alerts.",
 "Not a CLIP failure. <b>'dustbin' was absent from the detector's class list</b> while 'petrol can' "
 "was present, so the nearest available class won — and the trust threshold of 0.15 meant that "
 "label overrode CLIP entirely.",
 "Added benign lookalikes for every weapon class; raised the trust threshold to 0.35.",
 "An open-vocabulary class list is a forced choice. Whatever you leave out is assigned to "
 "whatever you left in.")

bug("3.4","Margins of 0.03 — the checks were never working",
 "Alerts were either constant or absent; no threshold satisfied both.",
 "Four of five negative phrases for [fire] contained the word 'fire'. <b>CLIP has no "
 "representation of negation</b> — 'a scene with no fire' embeds close to 'fire', so the system "
 "was subtracting the thing it was looking for.",
 "Rewrote every negative to describe what IS present. Margins doubled to tripled.",
 "Know what your model cannot represent. Negation is a classic blind spot in contrastive models.")

bug("3.5","Constrained decoding produced clean, wrong answers",
 "Forcing a VLM to answer only 'yes' or 'no' returned 'no' to everything.",
 "The token bitmask collapsed the distribution rather than steering it — including on questions "
 "the model answered correctly when unconstrained.",
 "Free-form generation with a three-way parser; unreadable answers return null and are dropped.",
 "Clean output format is not correctness. A constraint that improves parsing can destroy content.")
E.append(PageBreak())

# ============ 4 DISTILLATION ============
E.append(Paragraph("4 · Knowledge distillation",H1))
E.append(Paragraph(
 "The encoder was 93M of a 100M budget — 85% of the cost for one component. SigLIP2 carries "
 "capacity for artwork, food and celebrities; this system sees streets, gates and vehicles.",BODY))
E.append(img("distill.png"))
E.append(Paragraph("Figure 3 — The teacher's own embeddings are the training targets, so no human "
                   "labels are involved and every image is usable training data.",CAP))
E.append(Paragraph("Three decisions that made it work",H3))
E.append(tbl([
 ["Decision","Reasoning"],
 ["Inherit, never start from scratch",
  "Random initialisation needs millions of images; a pretrained small tower needs tens of thousands."],
 ["Replace only the image tower",
  "The student is trained INTO the teacher's space, so all cached text embeddings, thresholds and "
  "vocabularies keep working. Swap the encoder, keep the system."],
 ["Optimise what the pipeline reads",
  "Nothing downstream consumes a raw embedding — it consumes margins between near-identical "
  "phrases. A student can hit 0.97 agreement and still flatten those margins to nothing."],
],[48*mm,117*mm],fs=8.5))
E.append(Spacer(1,7))
E.append(img("params.png"))
E.append(Paragraph("Figure 4 — Detector size was never the problem; the encoder was.",CAP))
E.append(Spacer(1,4))
E.append(callout("The measurement that mattered",
 "Cosine agreement with the teacher was 0.881, which sounded mediocre. But probe accuracy — what "
 "the system actually does — came within ~3%. <b>Agreement measured similarity; AP measured "
 "whether it still works.</b> A probe is fitted to whatever geometry the encoder produces, so it "
 "needs the classes to stay separable, not the embeddings to be identical.",GOOD,"#ecfdf5"))
E.append(PageBreak())

# ============ 5 WEAK SUPERVISION ============
E.append(Paragraph("5 · Creating labels where none existed",H1))
E.append(Paragraph(
 "The project had no labelled evaluation data of any kind. Every threshold was set by intuition "
 "and adjusted by looking at single screenshots — which is how the same parameter got moved in "
 "both directions in one day.",BODY))
E.append(img("labelling.png"))
E.append(Paragraph("Figure 5 — UCF-Crime provides 1,900 real CCTV videos labelled by category. "
                   "A vision-language model converts those video-level labels into frame-level ones.",CAP))
E.append(Paragraph("Why the non-events are the valuable part",H3))
E.append(Paragraph(
 "A four-minute Fighting video contains perhaps fifteen seconds of fighting. Frames from that "
 "video showing an empty corridor are not waste — they are <b>hard negatives</b>: same camera, "
 "same lighting, same place, no fight. A probe trained against those learns the event. Trained "
 "only against unrelated normal footage, it could learn 'grainy indoor corridor' instead.",BODY))
E.append(img("funnel.png"))
E.append(Paragraph("Figure 6 — 5.9% of answers were unparseable. Those were written as null and "
                   "excluded, never guessed: a wrong label is worse than a missing one.",CAP))
E.append(Spacer(1,3))
E.append(tbl([
 ["Engineering detail","Value","Why it mattered"],
 ["Batched inference","43 q/s vs ~1 q/s","one-at-a-time discards the whole point of vLLM"],
 ["Warmup pass","—","JIT compilation on first inference returned empty answers"],
 ["Resume-safe output","—","a 20-minute job that cannot resume is a 20-minute job you rerun"],
 ["Null on unparseable","5.9% dropped","silent misparsing would have poisoned every probe"],
],[38*mm,28*mm,99*mm],fs=8))
E.append(PageBreak())

# ============ 6 RESULTS ============
E.append(Paragraph("6 · Results",H1))
E.append(Paragraph("From hand-written phrases to fitted vectors",H3))
E.append(Paragraph(
 "A check was a comparison against sentences someone wrote. A probe is a single 768-number vector "
 "fitted to real examples — same dot product, same speed, same inspectability, but optimised for "
 "the decision instead of hoping a sentence lands in the right place.",BODY))
E.append(img("ap.png"))
E.append(Paragraph("Figure 7 — Average Precision on held-out frames. The baseline is the positive "
                   "rate, not 50%: with 2.6% positives, always answering 'no' scores 97% accuracy "
                   "and detects nothing.",CAP))
E.append(tbl([
 ["","before (phrases)","after (probes)"],
 ["separation, positives vs negatives","0.02 – 0.05 cosine","0.62 – 0.87 probability"],
 ["thresholds","guessed, moved twice in one day","measured at a false-alarm budget"],
 ["checks usable","unclear — no measurement existed","8 of 9, with AP stated"],
],[62*mm,50*mm,53*mm],fs=8.5))
E.append(Spacer(1,7))
E.append(callout("The check that failed, and why that is a good outcome",
 "<b>weapon_scene reached AP 0.34</b> — only 246 positives out of 9,289, because weapons are small "
 "and rarely legible in CCTV. It was rejected automatically by an AP gate and kept its phrase pair. "
 "A system that reports which of its components do not work is more useful than one that reports "
 "uniform success.",BAD,"#fef2f2"))
E.append(Spacer(1,7))
E.append(img("margins.png"))
E.append(Paragraph("Figure 8 — The effect of removing negation from negative phrases, measured on "
                   "the same 800 held-out crops before and after.",CAP))
E.append(PageBreak())

# ============ 7 CONCEPTS ============
E.append(Paragraph("7 · Concepts used, and what they mean",H1))
E.append(Paragraph("Written as an internship reference — the ideas worth carrying to the next project.",LEAD))

def cg(title,rows):
    E.append(Paragraph(title,H3))
    E.append(tbl([["Term","Meaning in this project"]]+rows,[42*mm,123*mm],fs=8.3))
    E.append(Spacer(1,5))

cg("Vision–language models",[
 ["CLIP / SigLIP","Two encoders trained so an image and its caption land near each other. Lets you classify by describing classes in words instead of training on examples."],
 ["Zero-shot","Using a model on a task it was never trained for by describing the task. Convenient, and weak on compositional questions — 'holding a knife' vs 'holding a phone' differ by one noun in otherwise identical sentences."],
 ["Embedding space","The shared coordinate system. Two encoders are only comparable inside the same one, which is why every cache here is stamped with its encoder identity."],
 ["Prompt ensembling","Averaging several wordings of the same concept to cancel the noise in a single text embedding."],
])
cg("Making decisions from similarity",[
 ["Cosine margin","best-positive minus best-negative similarity. This project's lived in a 0.02–0.05 band — the reason nothing worked until it was measured."],
 ["Temperature","Controls how sharply small differences become confident percentages. SigLIP's native value (~100) turns noise into 99% certainty."],
 ["Absolute floor","A minimum raw similarity below which the system declines to answer. The fix for three separate bugs."],
 ["Gating","Refusing to run a check unless its preconditions hold — no helmet check without a motorcycle."],
])
cg("Supervised learning on frozen features",[
 ["Linear probe","One vector fitted on top of a frozen encoder. Cheap, inspectable, and dramatically better than hand-written phrases."],
 ["Class imbalance","With 2.6% positives, 'always no' is 97% accurate. Handled with pos_weight in the loss and measured with Average Precision."],
 ["Average Precision","Area under precision–recall. Baseline is the positive rate, not 50%. Cannot be gamed by predicting the majority class."],
 ["Hard negatives","Examples that look like the target but are not. Worth far more per sample than easy ones."],
 ["Held-out split","Fitting and judging on the same data measures memorisation. 20% was withheld throughout."],
])
cg("Model compression",[
 ["Knowledge distillation","Training a small model to imitate a large one. The teacher's outputs are the labels, so no annotation is needed."],
 ["Weight inheritance","Starting the student from a pretrained small model rather than randomly. Cuts the data requirement by an order of magnitude."],
 ["Projector","A small trained map from the student's space into the teacher's. Because the backbone stays frozen, features are computed once and training takes minutes."],
 ["Task-aware loss","Optimising the quantity the system consumes — here, the similarity distribution over the project's own vocabulary — rather than generic embedding match."],
])
cg("Weak supervision",[
 ["Weak / noisy labels","Labels that are correct at a coarse level (this video contains a fight) but not at the level you need (which frames)."],
 ["VLM as annotator","Using a vision-language model offline to convert coarse labels into fine ones. It runs once and is not part of the deployed system."],
 ["Label noise handling","Unparseable answers written as null and excluded, rather than defaulted. A wrong label is worse than a missing one."],
])
cg("Systems and engineering",[
 ["CUDA compatibility","Driver version caps which builds run. A forward-compatibility library resolved a full day of blocked work — worth checking before rebuilding environments."],
 ["Environment isolation","The labelling stack lived in its own venv and communicated by JSONL, so it could not break the working one."],
 ["Batched inference","43 q/s vs ~1 q/s for the same model. Throughput is an architectural choice, not a tuning detail."],
 ["Caching with identity stamps","Every cached artefact records which encoder produced it and refuses to load against another."],
 ["Regression testing","Freezing outputs and diffing after a change — the only way to notice that fixing one thing broke another."],
])
E.append(PageBreak())

# ============ 8 REFLECTION ============
E.append(Paragraph("8 · What this project actually taught",H1))
E.append(Paragraph("Measure before tuning",H3))
E.append(Paragraph(
 "A full day went into adjusting thresholds by looking at individual screenshots. The same "
 "parameter was moved in both directions within hours. The underlying problem was that the signal "
 "being thresholded was 0.03 wide and nobody had measured it. <b>When tuning is not converging, "
 "the parameter is usually not the problem.</b>",BODY))
E.append(Paragraph("Distinguish ranking from evidence",H3))
E.append(Paragraph(
 "Softmax answers 'which is most likely'. It never answers 'is any of this true'. Three separate "
 "bugs — scene, naming, attributes — were the same mistake in different places.",BODY))
E.append(Paragraph("Failure modes matter more than accuracy",H3))
E.append(Paragraph(
 "Constrained decoding produced perfectly formatted answers that were wrong. A silent parsing bug "
 "would have marked every frame negative and looked like a data problem. <b>Prefer systems that "
 "fail loudly.</b>",BODY))
E.append(Paragraph("Negative results are results",H3))
E.append(Paragraph(
 "The RPN was dropped, weapon_scene was rejected on measurement, and the class-agnostic proposal "
 "path was abandoned. Each of those decisions was backed by numbers, and each saved more time "
 "than it cost.",BODY))
E.append(Paragraph("Architecture can compensate for a weaker model",H3))
E.append(Paragraph(
 "A 13.4M encoder performs nearly as well as a 93M one here, largely because the system trusts "
 "detector labels over zero-shot naming, uses grouped softmax for attributes, and gates checks on "
 "preconditions. A naive CLIP-only captioner would have collapsed at that size.",BODY))

E.append(Paragraph("What remains",H2))
E.append(tbl([
 ["Item","Why it matters","State"],
 ["Person-crop probes","All 8 person checks are still phrase-based; weapon_held misfires on children","scripts written"],
 ["Recall measurement","False-alarm rate is measured; catch rate is not","data ready"],
 ["Attribute floors","Same ranking-as-evidence bug, third location","identified"],
 ["Regression test set","94 prompts written for generated test images","unused"],
 ["weapon_scene","Needs more visible-weapon positives than CCTV provides","open"],
],[40*mm,85*mm,40*mm],fs=8.3))
E.append(Spacer(1,10))
E.append(callout("Closing note",
 "The system began the project with alerts that fired on everything or nothing, thresholds nobody "
 "could justify, and no way to tell whether a change helped. It ends with eight checks whose "
 "accuracy is stated as a number, thresholds derived from real footage, an encoder a seventh the "
 "size for a measured 3% cost, and a labelled dataset built from scratch. <b>The largest single "
 "improvement came from replacing sentences a human wrote with vectors fitted to data.</b>"))

doc=SimpleDocTemplate("/sessions/blissful-busy-euler/mnt/outputs/JANA2_Internship_Report.pdf",
    pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,topMargin=20*mm,bottomMargin=20*mm,
    title="JANA2 — Internship Report",author="Neeraj Tiwari")
doc.build(E,onFirstPage=footer,onLaterPages=footer)
print("PDF built")
