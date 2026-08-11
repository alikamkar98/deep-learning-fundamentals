# 🧠 Deep Learning Fundamentals — A Visual, Hands-On Notebook

A from-first-principles tour of how a neural network actually learns, built in **PyTorch** and taught through **visualizations**. Every core concept is not just explained but *shown* — you watch weights nudge, loss surfaces bend, curves fit, and models overfit and recover.

The goal isn't to memorize terms. It's to build real intuition for the machinery of training.

---

## What you'll learn (and *see*)

The notebook walks through each idea with a picture, not just a definition:

| # | Concept | How it's visualized |
|---|---------|---------------------|
| 1 | **Train / Validation / Test split** | Three-color scatter of the same curve, plus *why* each split exists |
| 2 | **Forward pass** | Layer-by-layer shape trace through the network |
| 3 | **Loss computation (MSE)** | 1D "nudge one weight → watch the loss move" curve + a 3D loss surface |
| 4 | **Optimizer step (gradient descent)** | A ball rolling downhill on the real loss curve, step by step |
| 5 | **Batch** | Dataset sliced into mini-batches, drawn to scale |
| 6 | **Iteration** | One weight update = one batch, counted live |
| 7 | **Epoch** | One full pass over the data; loss curves across epochs |
| 8 | **Regularization (L2 / weight decay)** | Overfit curve vs. regularized curve, side by side |
| 9 | **Dropout** | Neurons randomly switched off, and the effect on the fit |
| 10 | **Weight initialization** | Different init schemes and how they change where training starts |
| 11 | **Hyperparameter tuning** | A sweep over learning rate / size / regularization, ranked |

Plus a **real-data capstone** on the California Housing dataset to tie it all together (scaling, early stopping, test-set evaluation, R²).

---

## The two datasets

1. **A synthetic noisy curve** (`y = f(x) + noise`) — the main teacher. Because the input is 1-dimensional, you can *plot the exact function the network learns* and literally watch it fit, overfit, and respond to dropout and regularization.
2. **California Housing** (built into scikit-learn, no download) — a real 8-feature regression problem for the capstone, where the same techniques are applied to messy real-world data.

---

## Run it locally

```bash
# 1. (recommended) create a clean environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch
jupyter notebook notebooks/deep_learning_fundamentals.ipynb
```

Then run the cells top to bottom. Everything is CPU-friendly — no GPU required.

---

## Project layout

```
deep-learning-fundamentals/
├── notebooks/
│   └── deep_learning_fundamentals.ipynb   ← the whole course, start here
├── src/
│   └── viz.py                             ← small plotting helpers
├── requirements.txt
├── .gitignore
└── README.md
```

---

## A note on how to read it

Work through it slowly. Each section states the idea in one plain sentence, shows the math once, then spends most of its time on a picture you can poke at. When you see a code cell, try changing a number (a learning rate, a batch size, a dropout probability) and re-running — the whole point is to develop a feel for what each knob does.
