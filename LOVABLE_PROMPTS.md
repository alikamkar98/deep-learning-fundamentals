# 🎨 Lovable Prompts — showcase this project on your website

Copy-paste prompts to build a polished project page in [Lovable](https://lovable.dev) for **kamkar-ai.com**. They're written to match your existing **navy + amber** brand. Use them in order: the first builds the page, the rest refine it.

---

## ⭐ MAIN PROMPT — The interactive learning hub

This is the big one: an interactive page where each concept is a **clickable card** that opens
a panel teaching it deeply (what it is · why we chose it · the alternatives & how they differ ·
how we used it · the visualization). All the text lives in `CONCEPTS.md` and all the images are
public in the repo, so Lovable can pull both directly.

```
Build an interactive "learning hub" page for my portfolio called "Deep Learning Fundamentals".
Match my site's existing design system exactly — navy (#1f3a5f) and amber (#f59e0b), same fonts,
same rounded cards, fully responsive, works in light and dark.

The full teaching content is in this file — READ IT and use it as the source of truth for every
explanation, comparison table, and takeaway:
https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/CONCEPTS.md

PAGE STRUCTURE:

1) HERO — eyebrow "Interactive · PyTorch · Built from scratch"; headline "How a neural network
   actually learns"; a one-paragraph intro; and a row of 4 stat chips: "8 core concepts",
   "0.0755 test MSE", "1→16→16→1 network", "PyTorch from scratch".

2) "WHAT WAS THE DATA?" section — two cards (the synthetic curve, and California Housing), using
   the "The data" section of CONCEPTS.md, each with its image:
   - Synthetic curve:  https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/the_task.png
   - California Housing: https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/capstone.png

3) "THE EIGHT IDEAS" — a responsive grid of 8 CLICKABLE cards (4 cols desktop, 2 tablet, 1 mobile).
   Each card shows a number, an icon, the concept title, and its one-line tagline. Clicking a card
   opens a MODAL (or an expanding panel) for that concept containing, in this order:
     - the concept's visualization image (see URL list below)
     - "What it is"  (from CONCEPTS.md)
     - "Why we chose it"  (from CONCEPTS.md)
     - "The options & how they differ" — render the comparison table, and highlight the row marked
       "our choice" with an amber background + an "our choice" pill
     - "In our project"  (from CONCEPTS.md)
     - a highlighted "Takeaway" line
   The modal must close on the X button, on clicking the backdrop, and on the Escape key, and be
   keyboard accessible.

   The 8 concepts and their image URLs (base: .../main/assets/):
     1. Train / Validation / Test      → split_three_ways.png
     2. Forward Pass                    → architecture.png
     3. Loss & MSE                      → nudge_the_weight_loss.png
     4. Gradient Descent                → gradient_descent_ball.png
     5. Batch · Iteration · Epoch       → batch_epoch_diagram.png
     6. Regularization & Dropout        → overfit_vs_regularization_dropout.png
     7. Weight Initialization           → weight_init.png
     8. Hyperparameter Tuning           → hyperparameter_search.png

4) "HOW WE TRAINED" section — show the 4-step training loop (zero_grad → forward+loss → backward →
   optimizer.step) as a numbered list, plus the result line (test MSE 0.0755; California RMSE 0.512),
   next to this image:
   https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/watch_it_learn.png

5) FOOTER — a call-to-action button "Open the full notebook on GitHub" linking to
   https://github.com/alikamkar98/deep-learning-fundamentals

STYLE: use monospace for small labels/eyebrows (it nods to the Jupyter notebook origin), generous
whitespace, subtle hover lift on the cards, and a smooth fade/scale when a modal opens. Make it feel
like a premium interactive case study, not a blog post. Use the real content from CONCEPTS.md — do
not invent or summarize it away.
```

> **If Lovable can't fetch the CONCEPTS.md URL:** open `CONCEPTS.md` yourself, copy its contents,
> and paste them into the prompt where it references the file. Everything Lovable needs is in there.
>
> **Want to see the finished result first?** Open `learning-hub.html` from the repo in your browser
> (just double-click it) — it's a complete, working version of exactly this hub. You can even tell
> Lovable: *"rebuild this to match my site"* and paste the HTML.

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

## Prompt 2 — Add the visuals (real images, no upload needed)

The images live in this public repo, so Lovable can load them straight from these URLs —
no file upload required.

```
In the Deep Learning Fundamentals page, add a "Selected Visualizations" section between
"The Key Idea" band and the Tech Stack. Make it a responsive image gallery (3 columns on
desktop, 1 on mobile) with a clean card frame and soft shadow around each image, and a
short caption underneath each. Load the images directly from these URLs:

Image 1:
https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/nudge_the_weight_loss.png
Caption: "Nudge one weight → watch the loss change. The dashed tangent is the gradient."

Image 2:
https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/watch_it_learn.png
Caption: "Watch the curve get fit, epoch by epoch."

Image 3:
https://raw.githubusercontent.com/alikamkar98/deep-learning-fundamentals/main/assets/overfit_vs_regularization_dropout.png
Caption: "Overfitting vs. regularization vs. dropout, side by side."

Make each image responsive (max-width 100%, auto height) and lazy-loaded.
```

> Two bonus visuals you can add the same way if you want a richer gallery:
> - `.../main/assets/loss_surface_3d.png` — "The loss surface: learning is finding the lowest valley."
> - `.../main/assets/hyperparameter_search.png` — "Hyperparameter search, ranked by validation loss."

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
