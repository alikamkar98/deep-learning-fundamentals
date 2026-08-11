# 🎨 Lovable Prompts — showcase this project on your website

Copy-paste prompts to build a polished project page in [Lovable](https://lovable.dev) for **kamkar-ai.com**. They're written to match your existing **navy + amber** brand. Use them in order: the first builds the page, the rest refine it.

---

## Prompt 1 — Build the project showcase page (main prompt)

```
Add a new project showcase page to my portfolio site called "Deep Learning Fundamentals".
Match the existing site's design system exactly — same navy (#1f3a5f) and amber (#f59e0b)
palette, same fonts, same spacing, same rounded-card style, and keep it fully responsive
with a light/dark aware feel.

The page presents a hands-on notebook project where I built a neural network from the ground
up in PyTorch and taught every core concept through visualizations. Structure it as:

1. HERO SECTION
   - Eyebrow text: "Machine Learning · PyTorch · From Scratch"
   - Headline: "Deep Learning Fundamentals"
   - Subhead: "I built a neural network in PyTorch and visualized exactly how it learns —
     from a single weight nudging the loss, to overfitting and how to cure it."
   - Two buttons: a primary amber "View on GitHub" button and a secondary outline
     "Read the walkthrough" button that scrolls down.

2. "WHAT I BUILT" — a short 2-3 sentence intro paragraph in a centered narrow column.

3. CONCEPTS GRID — a responsive grid of cards (3 columns on desktop, 1 on mobile), each with
   a small icon, a bold title, and one line of description. Cards:
   - "Train / Val / Test" — the three datasets and why each one exists
   - "Forward Pass" — how an input flows through the network to a prediction
   - "Loss & MSE" — measuring how wrong the model is, one number
   - "Gradient Descent" — nudging each weight downhill on the loss curve
   - "Batch · Iteration · Epoch" — the three units of a training run
   - "Regularization & Dropout" — stopping the model from memorizing noise
   - "Weight Initialization" — where training starts from
   - "Hyperparameter Tuning" — searching for the best settings
   Give each card a subtle hover lift and an amber accent border on hover.

4. "THE KEY IDEA" — a highlighted callout band (navy background, white text) that says:
   "The heart of learning: nudge one weight, and watch the loss move. The slope of that
   curve is the gradient — the arrow that tells the optimizer which way to step."

5. TECH STACK — a horizontal row of small pill badges: PyTorch, NumPy, Matplotlib,
   scikit-learn, Jupyter, Pandas.

6. FOOTER CTA — a final centered call to action with the amber "View on GitHub" button.

Keep the tone confident and clean. Use generous whitespace. Make it look like a premium
case study, not a blog post.
```

> After running this, replace the two GitHub button links with your real repo URL.

---

## Prompt 2 — Add the visuals (screenshots)

```
In the Deep Learning Fundamentals page, add a "Selected Visualizations" section between
"The Key Idea" band and the Tech Stack. Make it an image gallery of 3 wide screenshots
with short captions underneath each:
- Image 1 caption: "Nudge one weight → watch the loss change. The dashed tangent is the gradient."
- Image 2 caption: "Watch the curve get fit, epoch by epoch."
- Image 3 caption: "Overfitting vs. regularization vs. dropout, side by side."
Use a clean card frame around each image with a soft shadow, and make the gallery a
responsive grid that stacks to one column on mobile. Use placeholder images for now.
```

> Then upload three screenshots from the notebook (right-click a chart → save, or screenshot):
> the **nudge-the-weight** loss curve, the **watch-it-learn** epochs plot, and the
> **overfitting/regularization/dropout** comparison. These three are the most visually striking.

---

## Prompt 3 — Polish & motion

```
Polish the Deep Learning Fundamentals page:
- Add subtle fade-in-on-scroll animations to each section as it enters the viewport.
- Make the concept cards animate in with a slight stagger.
- Add a thin amber progress bar at the very top of the page that fills as you scroll.
- Ensure strong color contrast for accessibility, and that all interactive elements have
  visible focus states.
- Double-check it looks great on a phone: no horizontal scroll, comfortable tap targets,
  readable font sizes.
```

---

## Prompt 4 — Link it into the site

```
Add "Deep Learning Fundamentals" as a new entry in my projects list / portfolio grid on the
home page, using the same card style as my other projects. Use a short blurb:
"A visual, from-scratch tour of how a neural network learns — built in PyTorch."
Tag it with: PyTorch, Deep Learning, Data Viz. Link the card to the new project page.
```

---

## Ready-to-use copy (paste anywhere you want to tweak text)

**One-line summary**
> A visual, from-scratch tour of how a neural network learns — built in PyTorch.

**Longer blurb**
> I wanted to truly understand deep learning, not just use it. So I built a neural network
> in PyTorch and turned every core idea into a picture: the three data splits, the forward
> pass, loss and gradients, batches and epochs, overfitting and its cures, and how to tune
> the whole thing. The result is a notebook you can run and poke at yourself.

**Skills demonstrated**
> PyTorch · neural network training · data visualization · model regularization ·
> hyperparameter tuning · reproducible ML workflows

---

### Tips
- Keep the navy/amber palette consistent with the rest of kamkar-ai.com so this page feels native.
- The three screenshots in Prompt 2 do the heavy lifting — they instantly communicate depth.
- If Lovable drifts from your brand, paste your existing color variables and font names and
  ask it to "use these exact tokens."
