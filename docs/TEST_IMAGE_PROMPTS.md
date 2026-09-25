# Test image generation prompts — JANA2

A regression set for Indian surveillance: roads, housing societies, malls,
factories. Every check gets a positive AND a hard negative, because the
false positives are what have actually hurt us.

---

## How to use

Every prompt is **BASE + SCENE**. The base is what makes the image look like
footage instead of a photograph, and it matters more than the scene text.

### BASE (prepend to every prompt)

```
CCTV security camera still, mounted high on a pole looking down at a steep
angle, wide angle lens with slight barrel distortion, available light only,
slightly soft focus, visible sensor grain and JPEG compression artifacts,
muted washed colours, India, documentary realism, no text overlay
```

### NEGATIVE PROMPT (for models that take one)

```
eye level, portrait, professional photography, studio lighting, shallow
depth of field, bokeh, cinematic, dramatic lighting, sharp macro detail,
poster, illustration, cartoon, 3d render, watermark, centred subject
```

### Why the base matters more than the scene

The encoder never sees a photograph. It sees a crop from a camera bolted
four metres up a pole. Two things follow:

- **Angle.** Generators default to eye level. A downward view changes the
  shape of everything — people become foreshortened, vehicles are seen from
  above, faces are rarely visible. Get this wrong and the test set tests the
  wrong distribution.
- **Degradation.** Real frames are grainy, compressed and often badly lit.
  A clean render is an easier image than anything the system will meet in
  service, so it would flatter the results.

---

## 1. NORMAL SCENES — generate the most of these

Most footage is boring. These are the false-alarm test, and they are worth
more than the event images: an alert system that fires on a normal Tuesday
is useless regardless of how well it catches fights.

```
1.  the entrance gate of a residential housing society, a boom barrier down,
    a guard in khaki standing beside a cabin, two parked scooters
2.  a narrow residential lane between apartment blocks, a few parked cars,
    a woman in a saree walking with a shopping bag
3.  a society basement car park, rows of parked cars under fluorescent tubes,
    nobody present
4.  an apartment corridor with closed flat doors and a lift lobby, empty
5.  a busy Indian city street at midday, auto rickshaws, motorcycles, a bus,
    pedestrians crossing
6.  a traffic junction with a signal, mixed traffic waiting, a traffic
    policeman at the side
7.  a shopping mall entrance with a metal detector and a bag scanner, a queue
    of shoppers
8.  a mall interior with an escalator and shopfronts, ordinary shoppers
9.  a factory floor with machinery, three workers in hard hats and safety
    vests going about their work
10. a warehouse interior, stacked cartons, a forklift, one worker
11. a residential society children's play area, children playing on a slide,
    two adults watching
12. a society gate at night lit by a single sodium lamp, a guard seated, one
    parked motorcycle
13. a wet road after rain, traffic moving normally, reflections on the tarmac
14. a street food vendor cart with customers standing around it
15. a rooftop water tank area of an apartment building, empty
16. an ATM kiosk with one person using the machine
17. a bus stop with people waiting, a bus approaching
18. a construction site with scaffolding and workers in helmets, materials
    stacked
```

---

## 2. EVENT SCENES — one per check

### fire / smoke

```
19. a parked car on fire on a residential street, orange flames and black
    smoke, people standing back at a distance
20. thick grey smoke pouring from a ground floor shop shutter, a small crowd
    gathering
21. a rubbish pile burning at the edge of a road, low flames, smoke drifting
```

### fight / aggressive

```
22. two men grappling and striking each other in the middle of a narrow
    street, a ring of onlookers
23. four men brawling outside a shop shutter at night
24. one man shoving another backwards against a compound wall
```

### person_down / on_ground

```
25. a man lying motionless face down on a road beside a fallen motorcycle
26. an elderly person collapsed on an apartment corridor floor, a walking
    stick nearby
27. a labourer lying on the ground at a construction site, hard hat beside him
```

### weapon_scene / weapon_held

```
28. a man raising a long curved blade towards another man on a street at night
29. a group of four men walking down a lane carrying iron rods and wooden
    sticks
30. a man pointing a small pistol, seen from a high camera angle, a second
    person backing away
```

### accident

```
31. two cars collided head on at a junction, crumpled bonnets, debris and
    broken glass scattered on the road
32. an auto rickshaw overturned on its side on a road, a small crowd around it
33. a motorcycle lying on the tarmac beside a car with a dented door, a rider
    sitting on the kerb
```

### climbing / intrusion

```
34. a man halfway over a society compound wall at night, legs still on the
    outside
35. a person scaling an iron gate topped with spikes
36. a man climbing a drainage pipe up the side of an apartment building
```

### snatching / robbery

```
37. a motorcycle pillion rider grabbing a handbag from a woman walking on a
    pavement
38. a man wrenching a chain from around another person's neck on a dark street
39. two men holding a shopkeeper against a counter while a third empties a
    cash drawer
```

### vandalism

```
40. a man smashing the windscreen of a parked car with a stone
41. two youths spraying paint across a compound wall
42. a broken shopfront window with glass scattered on the pavement
```

### vehicle_tampering

```
43. a man crouched beside a parked motorcycle at night working at the ignition
    lock with a tool
44. a person reaching in through the broken window of a parked car in a
    basement car park
```

### unattended_bag

```
45. a black rucksack sitting alone on the floor of a mall corridor, no one
    within several metres
46. a suitcase standing by itself against a wall at a railway platform
```

### no_helmet / overloaded / wrong_side

```
47. three people riding one motorcycle on a city road, none wearing helmets
48. a small truck piled far above its cab with sacks, leaning to one side
49. a motorcycle riding towards the camera against the direction of traffic
    on a divided road
```

### crowd_surge / loitering

```
50. a dangerously dense crowd pressed together at a narrow gateway, people
    pushing forward
51. six young men standing idle in a group on a street corner late at night
```

### flooding

```
52. a city road submerged under knee deep brown water, an auto rickshaw
    wading through it
53. a flooded society basement car park, cars standing in water up to their
    wheel arches
```

### littering

```
54. a man tossing a plastic bag of rubbish onto the roadside from a scooter
55. rubbish strewn across a pavement beside an overflowing bin
```

### unattended_child

```
56. a small child alone at the edge of a busy road, no adult nearby
57. a toddler standing by itself at an open society gate
```

### darkness / obstructed_camera

```
58. an almost entirely black night scene, faint outline of a gate barely
    visible
59. a camera view mostly blocked by a leaf pressed against the lens
60. a camera view smeared and blurred by rain and dirt on the housing
```

---

## 3. HARD NEGATIVES — the most valuable images in the set

Every one of these produced a real false positive today, or is one wording
away from doing so. If the system stays quiet on these, it is trustworthy.
If it alerts, you know exactly which check to fix.

```
61. a dusty weathered old car with faded paint and a dented bumper parked at
    a kerb            -> must NOT read as [accident]
62. a car raised on a ramp in a repair garage, panels removed
                      -> must NOT read as [accident]
63. cars queued bumper to bumper in slow traffic
                      -> must NOT read as [accident]
64. a security guard standing at a gate holding a long bamboo lathi
                      -> must NOT read as [weapon_held]
65. a street vendor chopping vegetables with a large knife at a roadside cart
                      -> must NOT read as [weapon_held]
66. a workman carrying a hammer and a length of pipe across a construction
    site                -> must NOT read as [weapon_held]
67. a man walking in the rain holding a closed black umbrella at his side
                      -> must NOT read as [weapon_held]
68. a person asleep on a public bench, arm over their face
                      -> must NOT read as [person_down]
69. a mechanic lying on his back under a car
                      -> must NOT read as [person_down]
70. a man squatting on his heels at the roadside drinking tea
                      -> must NOT read as [person_down]
71. an orange and red sunset sky over an apartment block
                      -> must NOT read as [fire]
72. bright sodium street lamps glowing orange in the dark
                      -> must NOT read as [fire]
73. a woman in a bright orange saree walking down a lane
                      -> must NOT read as [fire]
74. thick morning fog lying over a residential road
                      -> must NOT read as [smoke]
75. dust thrown up behind a truck on an unpaved road
                      -> must NOT read as [smoke]
76. steam rising from a roadside tea stall
                      -> must NOT read as [smoke]
77. two men shaking hands and embracing in greeting on a street
                      -> must NOT read as [fight]
78. a crowd of people queuing in an orderly line at a counter
                      -> must NOT read as [crowd_surge] or [loitering_group]
79. a family walking together through a mall
                      -> must NOT read as [loitering_group]
80. a person handing a bag politely to another person
                      -> must NOT read as [snatching]
81. a shopkeeper passing a parcel across a counter to a customer
                      -> must NOT read as [snatching]
82. a driver unlocking his own car door with a key
                      -> must NOT read as [vehicle_tampering]
83. a man loading suitcases into a car boot
                      -> must NOT read as [vehicle_tampering]
84. a colourful painted mural covering a compound wall
                      -> must NOT read as [vandalism]
85. a workman fitting a new pane into a window frame
                      -> must NOT read as [vandalism]
86. a traveller wheeling a suitcase along a mall corridor
                      -> must NOT read as [unattended_bag]
87. a backpack worn on a student's shoulders
                      -> must NOT read as [unattended_bag]
88. a child holding an adult's hand at a society gate
                      -> must NOT read as [unattended_child]
89. shallow puddles on a road after light rain
                      -> must NOT read as [flooding]
90. rubbish being dropped correctly into a dustbin
                      -> must NOT read as [littering]
91. a dimly lit corridor with fittings still clearly visible
                      -> must NOT read as [darkness]
92. a slightly out of focus but usable camera view
                      -> must NOT read as [obstructed_camera]
93. a rider wearing a full face helmet on a motorcycle
                      -> must NOT read as [no_helmet]
94. an empty parked motorcycle with a helmet resting on the seat
                      -> must NOT read as [no_helmet]
```

---

## 4. DIFFICULTY MODIFIERS

Add one or two of these to any prompt above. The system should degrade
gracefully, not confidently — and easy images will not tell you whether it
does.

```
at night lit only by a single street lamp
in heavy monsoon rain, water on the lens
strong backlight with the sun behind the subject, silhouetted figures
the subject small and far away in the upper part of the frame
the subject partly hidden behind a parked vehicle
motion blurred, the subject moving quickly
crowded frame with many people and vehicles at once
harsh midday shadows across the scene
low resolution, heavily compressed, blocky artifacts
infrared night vision, monochrome green tint
camera angle very steep, almost directly overhead
```

Particularly worth generating: **the small-and-far-away variant**. Your
crops of distant objects are often 40 pixels across, and a check that works
on a large clear subject may collapse entirely at that size. That is the
regime real footage lives in.

---

## 5. How to organise the output

```
data/testset/
  normal/         1-18  and every hard negative 61-94
  events/         19-60
  events_hard/    events combined with difficulty modifiers
```

That layout feeds the existing tools directly:

```bash
python -m scripts.caliberate --normal data/testset/normal \
                             --events data/testset/events --budget 2
python -m scripts.batch_report --images data/testset/events
python -m scripts.snapshot save --images data/testset --tag baseline
```

`caliberate` with both folders reports the false-alarm rate on normals AND
what those thresholds would catch on events — the two-sided measurement we
have never had.

---

## 6. One honest caveat

Generated images are a **regression set, not a validation set**.

They tell you whether a change broke something, and they let you probe
specific failures on demand — which is genuinely valuable, and you have
neither of those things today.

They do not tell you the system works on real footage. Synthetic images
carry their own systematic biases: a generator's idea of a CCTV frame is
smoother, better composed and more legible than an actual one, and a model
tuned on them can look excellent and still fail on a real gate camera at
dusk.

So use these to iterate quickly, and treat real footage — UCF-Crime, or
recordings from actual sites — as the thing that decides whether it works.
