# Concepts — the deep explanations

A plain-language study reference for every idea in the project. For each: what it is, why we chose it over the alternatives, and how it showed up in our code. Interactive version: `learning-hub.html`. Notebook: [https://github.com/alikamkar98/deep-learning-fundamentals](https://github.com/alikamkar98/deep-learning-fundamentals).

## The data

### The synthetic curve (main teacher)

A noisy 1-D curve: **y = sin(1.6x) + 0.35x + noise**, with Gaussian noise (σ = 0.25). Because the input is a single number, we can **plot the exact function the network learns** and watch it fit, overfit, and respond to every technique. That visibility is the whole reason we start here.

### California Housing (real-data capstone)

A real dataset built into scikit-learn: **20,640 districts, 8 features** (median income, house age, rooms, location…) → predict median house value. Messier and higher-dimensional, so we add real-world habits: **feature scaling** (fit on train only) and **early stopping**. Result: test **RMSE ≈ 0.512**.

## The eight concepts

### 01 · Train / Validation / Test

*Three separate datasets — and why each one has to exist.*

**What it is** — We chop the data into three groups that never overlap. The **train** set is what the optimizer fits the weights on. The **validation** set is what *we* look at between experiments to tune settings and decide when to stop. The **test** set is sealed away and opened exactly once, at the very end, for a single honest score.

**Why we chose it** — If you judge a model on the same data it learned from, you're measuring memorization, not understanding — it has already seen the answers. Validation gives us a fair signal to tune against *without* burning the test set. Test stays untouched so the final number can't be gamed. The golden rule: **never let information from test leak into training.**

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Train** | Model sees it every single step; weights are fit to it | **← our choice** |
| **Validation** | We see it between runs — tunes hyperparameters + early stopping | **← our choice** |
| **Test** | Nobody sees it until the end — one unbiased final score | **← our choice** |
| **Cross-validation** | Alternative when data is scarce: rotate which slice is val |  |

**In our project** — 240 points → **168 train (70%)**, **36 validation (15%)**, **36 test (15%)**. We computed the test MSE only once, at the very end: **0.0755**.

> **Takeaway:** Fit on train · tune on validation · score once on test.

### 02 · Forward Pass

*How an input flows through the network into a prediction.*

**What it is** — The forward pass pushes an input `x` through the layers to get a prediction `ŷ`. Each **Linear** layer computes a weighted sum `weight · input + bias`. Between them sits a **nonlinearity** (we used **Tanh**) that bends the signal.

**Why we chose it** — Without a nonlinearity, stacking linear layers just collapses into a single straight line — it could never fit our wiggly curve. The nonlinearity is exactly what lets the network *bend*. We picked **Tanh** because it's smooth and symmetric, which suits fitting a smooth regression curve.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Tanh** | Smooth, output in [-1, 1]; great for smooth regression — our pick | **← our choice** |
| **ReLU** | max(0, x): fast, sparse, the default for deep/vision nets |  |
| **Sigmoid** | Squashes to (0, 1); saturates and can stall gradients |  |
| **GELU** | Smooth ReLU-like curve; common in modern transformers |  |

**In our project** — A **1 → 16 → 16 → 1** MLP. A batch of 5 inputs enters as shape (5, 1) and stays (5, 1) all the way to 5 predictions — the whole batch flows through at once.

> **Takeaway:** The forward pass is the network's guess; everything else exists to improve it.

### 03 · Loss & MSE

*Turning 'how wrong is the model' into a single number.*

**What it is** — A **loss** collapses the model's wrongness into one number to minimize. For regression we used **Mean Squared Error**: take each prediction's gap from the target, **square it**, and average. Lower is better; 0 is perfect.

**Why we chose it** — Squaring does two useful things: it makes over- and under-shooting both count as positive, and it punishes big misses *far* more than small ones. It's also smooth and differentiable, which gives clean gradients for learning. And it's the *natural* loss when your noise is Gaussian — which is precisely how we generated the data. That's why MSE, not something else.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **MSE (L2)** | Squares errors — smooth, punishes big misses, our regression pick | **← our choice** |
| **MAE (L1)** | Absolute error — more robust to outliers, but harsher gradients |  |
| **Huber** | MSE near zero, MAE far out — a blend for outlier-heavy data |  |
| **Cross-Entropy** | For classification, not regression — compares probabilities |  |

**In our project** — `nn.MSELoss()`. Final train loss **0.054**; final test MSE **0.0755** — essentially the noise floor (the noise variance is 0.25² = 0.0625), so the model recovered the true curve about as well as is possible.

> **Takeaway:** The loss defines what 'good' means — choose it to match your task.

### 04 · Gradient Descent

*Nudging every weight downhill on the loss curve.*

**What it is** — The **gradient** is the slope of the loss with respect to a weight — it says which way, and how steeply, the loss changes. Gradient descent steps **opposite** the slope: `w ← w − learning_rate · gradient`. Backprop computes this slope for every weight at once.

**Why we chose it** — The gradient points *uphill*; to reduce the loss we step downhill. The **learning rate** is the step size and it's the single most important knob: too small and training crawls, too big and it overshoots the valley or diverges entirely. We used the **Adam** optimizer, which adapts the step size per-weight automatically and is a robust default.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Adam** | Adaptive per-weight step size; robust, fast default — our pick | **← our choice** |
| **SGD** | Plain gradient descent; simple but needs careful lr tuning |  |
| **SGD + Momentum** | Accumulates velocity to power through flat spots |  |
| **RMSProp** | Adaptive like Adam's ancestor; good for noisy objectives |  |

**In our project** — Adam at **lr = 0.01**. We visualized a single weight literally **rolling down** the loss curve one step at a time, and showed the same landscape as a 3-D surface — learning is the search for its lowest valley.

> **Takeaway:** Learning is just many small, repeated downhill steps.

### 05 · Batch · Iteration · Epoch

*The three units that measure a training run.*

**What it is** — A **batch** is a small handful of examples processed together. An **iteration** is one weight update — one batch in, one step. An **epoch** is one full pass over the *entire* training set. So iterations-per-epoch = train size ÷ batch size.

**Why we chose it** — We rarely use the whole dataset in one shot: full-batch updates are slow and memory-hungry, while one-example-at-a-time is very noisy. **Mini-batches** are the sweet spot — fast, memory-friendly, and the little bit of noise they add actually helps the model generalize.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Mini-batch** | A chunk per step — the standard compromise, our choice | **← our choice** |
| **Full-batch** | All data every step — stable gradient but slow, memory-heavy |  |
| **Stochastic (1)** | One example per step — very noisy, but fast and escapes ruts |  |

**In our project** — **168** train points with **batch size 16** → **11 iterations per epoch**, run for **250 epochs**. So the weights were updated 11 × 250 ≈ 2,750 times.

> **Takeaway:** Epoch counts passes · iteration counts updates · batch sets the chunk.

### 06 · Regularization & Dropout

*Stopping the model from memorizing noise.*

**What it is** — **Overfitting** is when a model memorizes the training noise instead of the pattern — its train loss keeps falling while its **validation loss climbs**. Regularization fights this. **L2 (weight decay)** penalizes large weights so the function stays smooth. **Dropout** randomly switches off neurons during training so the network can't lean on any single one.

**Why we chose it** — A model with lots of capacity on little data will happily thread every noisy dot — and then fail on anything new. L2 pulls weights toward zero for smoother fits; dropout forces redundant, robust features (like an ensemble of smaller networks). Early stopping simply keeps the weights from the best validation epoch.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **L2 / weight decay** | Shrinks weights → smoother model; our main regularizer | **← our choice** |
| **Dropout** | Randomly drops neurons in training → robustness; also used | **← our choice** |
| **Early stopping** | Keep the best-validation weights, stop early; used in capstone | **← our choice** |
| **L1** | Pushes weights to exactly zero → sparse feature selection |  |

**In our project** — We deliberately over-fit a big model on just **25 points**, then cured it: **L2 (1e-2)** and **dropout (0.25)** both cut validation loss and smoothed the curve. The capstone combined **dropout 0.1 + weight decay 1e-4 + early stopping** (stopped at epoch 81).

> **Takeaway:** A model that memorizes the training set fails on new data — regularization buys generalization.

### 07 · Weight Initialization

*Where training starts from — before any learning happens.*

**What it is** — Every weight must start at *some* value. That starting point decides whether training can even begin. **Zeros** make every neuron identical (they get the same gradient forever and can never differentiate). **Too large** makes outputs explode and gradients misbehave. **Xavier/Glorot** init keeps the signal a sensible size as it flows through.

**Why we chose it** — Good initialization doesn't solve the problem — it gives the optimizer a *fair starting line*. Keeping activations and gradients at a stable magnitude across layers is what lets learning proceed smoothly instead of stalling or blowing up.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Xavier / Glorot** | Scaled for tanh/sigmoid — keeps signal stable; our pick | **← our choice** |
| **He / Kaiming** | Scaled for ReLU networks specifically |  |
| **Small random** | Works, but not tuned to the layer sizes |  |
| **Zeros** | Broken: symmetry means neurons never differentiate |  |

**In our project** — We showed all three side by side on an untrained network: **zeros** gave a dead flat line, **too big** gave a wild saturated shape, and **Xavier** gave a gentle wiggle ready to be shaped by training.

> **Takeaway:** A fair starting line matters — initialization is not an afterthought.

### 08 · Hyperparameter Tuning

*Searching for the settings you can't learn by gradient.*

**What it is** — **Weights** are learned by the optimizer. **Hyperparameters** — learning rate, hidden size, weight decay, dropout, batch size, epochs — are chosen by *you*, before training. There's no gradient for them, so we find good ones by **searching** and ranking by validation loss.

**Why we chose it** — You can't backprop through 'how many hidden units'. So we try many configurations, score each on **validation** (never test), and pick the winner — then confirm it once on test. We used a **grid search**: exhaustive over a small set, simple and easy to reason about.

**The options & how they differ**

| Option | What it does | |
|---|---|---|
| **Grid search** | Try every combination — simple, thorough; our method | **← our choice** |
| **Random search** | Sample combinations — often more efficient per run |  |
| **Bayesian / Optuna** | Model which settings look promising — sample-efficient |  |
| **Manual** | Hand-tune by intuition — fast but easy to fool yourself |  |

**In our project** — A grid over **lr {0.001, 0.01, 0.05} × hidden {8, 32} × weight_decay {0, 1e-3}** = 12 configs. Winner: **lr 0.01, hidden 8, weight_decay 1e-3** (val loss 0.037). Only then did we score the test set.

> **Takeaway:** Tune on validation, rank the candidates, confirm once on test.

## How we trained

Every epoch, for every batch: `zero_grad()` → forward + loss → `backward()` → `optimizer.step()`. The toy curve reached test MSE **0.0755** (≈ the noise floor); the same pipeline transferred to California Housing at **RMSE 0.512**.
