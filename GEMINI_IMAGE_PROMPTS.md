# 🖼️ Gemini image-generation prompts — visuals for the learning hub

Prompts to generate **beautiful, on-brand illustrations** for each concept (for Gemini's image
generation, e.g. Imagen / "Nano Banana"). They share one art direction so the whole set looks
like it belongs together.

### How to use
1. Generate them **in one session**, in order, and tell Gemini *"keep the exact same art style,
   palette, and framing as the previous image"* so the set stays consistent.
2. Use a consistent aspect ratio — **16:9** for the hero, **4:3 or square** for concept cards.
3. Download them, then either upload to Lovable, or drop them in the repo's `assets/` folder and
   reuse the same URL pattern.

> **Note:** image models are great at *aesthetic, conceptual* illustrations but bad at exact
> charts, numbers, and long text. So these prompts ask for **visual metaphors**, not precise
> graphs. Keep the real matplotlib plots (in `assets/`) for anything that must be numerically
> accurate; use these Gemini images to make the page *look* premium.

---

## The shared style (prepend this to every prompt)

```
A minimal, modern flat-vector editorial illustration for a premium tech/education website.
Deep navy background (#0f2740), a single warm amber accent (#f59e0b), soft off-white line work,
clean geometric shapes, generous negative space, subtle grain, gentle glow. Premium, calm,
intelligent feel — not busy, not cartoonish, no photorealism. Do NOT render paragraphs of text.
Subject:
```

---

## Hero (16:9)
```
...an abstract neural network gracefully learning a smooth curve — glowing amber nodes connected
by thin off-white lines on the left, resolving into a single elegant amber sine-like curve on the
right, as if the network is "drawing" the curve. Sense of intelligence and motion. Cinematic, wide.
```

## 1 · Train / Validation / Test (square)
```
...a single stream of small dots being sorted into three separate rounded trays of different
sizes: one large tray (most dots), one medium, one small tray set slightly apart and glowing
amber (kept sealed for the end). Clean icons suggesting practice, check, and final exam.
```

## 2 · Forward Pass (4:3)
```
...a stylized left-to-right neural network of glowing nodes in layers; a single bright amber
pulse of light traveling forward from one input node, through the hidden layers, arriving at one
output node on the right. Directional, flowing, elegant.
```

## 3 · Loss & MSE (4:3)
```
...a smooth glowing valley-shaped curve on a navy grid; a single amber sphere resting on its slope
with a thin amber tangent arrow showing the downhill direction. Conveys "measuring how wrong, and
which way is better". Minimal, precise.
```

## 4 · Gradient Descent (4:3)
```
...an amber sphere rolling down into the lowest point of a smooth 3D navy valley/terrain, leaving
a dotted trail of its previous positions behind it. Sense of settling into the bottom. Depth, calm.
```

## 5 · Batch · Iteration · Epoch (4:3)
```
...a long horizontal row of small identical squares grouped into a few neat bundles, with one
circular amber arrow looping back over the whole row — suggesting small chunks processed in
repeated full passes. Rhythmic, orderly.
```

## 6 · Regularization & Dropout (4:3)
```
...a neural network where a few nodes are dimmed and switched off (marked with subtle amber x's),
beside a contrast of one jagged nervous line versus one calm smooth amber line. Conveys "stop
memorizing noise, stay smooth". Balanced composition.
```

## 7 · Weight Initialization (4:3)
```
...three small identical neural networks lined up at a glowing amber starting line like runners:
the first is flat and dark (stuck), the second chaotic and tangled, the third balanced and gently
glowing (ready). Conveys "where training starts matters".
```

## 8 · Hyperparameter Tuning (4:3)
```
...a sleek control panel of dials and sliders beside a small grid of option tiles, with exactly
one tile glowing bright amber as the chosen best setting. Conveys "searching for the right knobs".
Clean, precise, minimal.
```

---

### After you generate them
- Name them to match the concepts (e.g. `gen_loss.png`, `gen_gradient_descent.png`).
- In Lovable, either upload them, or replace the image URLs in the MAIN PROMPT (in
  `LOVABLE_PROMPTS.md`) with your new image URLs.
- Keep one visual language across the whole page — if one image looks off, regenerate it with
  *"match the style of the others exactly."*
