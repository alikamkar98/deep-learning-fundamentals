# 🧑‍🏫 The code, explained in plain language

This walks through **every part** of the notebook and explains, in everyday words, **what the code does** and **why it's there** — with the neural network explained in the most detail. It assumes you have **never written code**. If you read it slowly, top to bottom, you'll be able to explain the whole project to anyone.

---

## Part 0 — A 2-minute crash course in reading code

Before the notebook, five ideas that unlock *all* of it. Code is just a recipe written in a very strict language.

| Idea | In plain words | Everyday analogy |
|------|----------------|------------------|
| **Variable** | A labeled box that holds a value. `N = 240` means "put 240 in a box named N." | A jar with a label on it. |
| **Function** | A little machine: you put something in, it gives something back. | A blender: fruit in → smoothie out. |
| **Library (import)** | A pre-made toolbox someone else built, that we borrow. | Buying a power drill instead of forging one. |
| **Object & method** | A "thing" that also knows how to *do* actions. We use a dot: `model.forward(...)` = "hey model, do your forward action." | A dog (object) that can `.sit()` and `.bark()` (methods). |
| **Loop** | "Do this same step for each item in a list." | Stamping every envelope in a pile. |

Two more tiny things:
- A line starting with **`#`** is a **comment** — a human note the computer ignores.
- **Indentation** (the spaces at the start of a line) is how Python groups steps that belong together, like sub-bullets under a bullet.

That's genuinely enough to follow everything below.

---

## The whole notebook in one sentence

> We **invent** a curvy line, **hide it** under random noise, and then teach a tiny artificial "brain" to **rediscover** the hidden line just by looking at the noisy dots — and we watch every step of that learning happen. Then we repeat the same recipe on **real** house-price data to prove it wasn't a fluke.

Now, section by section.

---

## Part 1 — Setup: laying out our tools

```python
%matplotlib inline
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join("..", "src")))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from viz import set_style, draw_mlp, COLORS

set_style()
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
print("PyTorch", torch.__version__, "| device:", device)
```

**In plain words, line by line:**
- `%matplotlib inline` — "when you draw a chart, show it right here in the page."
- The `os.environ[...]` line — a **Windows-only safety fix**. On this computer two copies of a math engine can clash and crash; this line tells them to get along. (Harmless everywhere else.)
- The `sys.path.append(...)` line — "also look in the neighbouring `src` folder for our little helper file." That's how we later use `draw_mlp` (the tool that draws the network picture).
- The five `import` lines — we **borrow four toolboxes**:
  - `numpy` (nicknamed `np`) — fast number-crunching (arrays of numbers).
  - `matplotlib` (`plt`) — drawing charts.
  - `torch` — **PyTorch**, the deep-learning toolbox (builds the brain, does the learning).
  - `nn` — the part of PyTorch specifically for **n**eural **n**etworks.
  - `viz` — *our own* helper for consistent colors and the network diagram.
- `set_style()` — makes all our charts look consistent.
- `SEED = 42` and the two `seed(...)` lines — this is important and subtle. Computers make "random" numbers from a starting point called a *seed*. **Fixing the seed means the randomness is the same every time you run it** — so you get the *exact same* results and pictures. Reproducibility. (42 is just a traditional joke number.)
- `device = torch.device("cpu")` — "do the math on the normal processor" (no fancy graphics card needed).
- `print(...)` — shows which PyTorch version is running, so we know the tools loaded.

**Why this cell exists:** every project starts by unpacking its tools and setting the rules of the game (here: *be reproducible*). Nothing has been learned yet — we're just laying out the workbench.

---

## Part 2 — Inventing the data

```python
def true_function(x):
    return np.sin(1.6 * x) + 0.35 * x

N = 240
x_all = np.random.uniform(-3.0, 3.0, size=N)
y_all = true_function(x_all) + np.random.normal(0, 0.25, size=N)
```

**In plain words:**
- `def true_function(x):` — we **define a machine** called `true_function`. You feed it a number `x`, and it returns `sin(1.6·x) + 0.35·x`. This is the **hidden pattern** — a gentle S-shaped wave that also drifts upward. *The network will never be told this formula.* It's the secret we want it to rediscover.
- `N = 240` — we'll make 240 example points.
- `x_all = np.random.uniform(-3.0, 3.0, size=N)` — pick 240 random `x` positions, spread evenly between −3 and 3. These are the "where we took measurements."
- `y_all = true_function(x_all) + np.random.normal(0, 0.25, size=N)` — for each `x`, compute the true `y`, then **add a little random wobble** (`normal(0, 0.25)` = small random nudges averaging zero). This wobble is **noise** — it mimics how real-world measurements are never perfectly clean.

**Why:** real data is *signal + noise*. By building our own data this way, we know the true answer, so later we can check how close the network got. The rest of this cell just **draws** the smooth true curve and scatters the noisy dots on top, so you can *see* the challenge.

---

## Part 3 — Splitting into train / validation / test

```python
from sklearn.model_selection import train_test_split
x_tmp, x_test, y_tmp, y_test = train_test_split(x_all, y_all, test_size=0.15, random_state=SEED)
x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=0.1765, random_state=SEED)
```

**In plain words:**
- We borrow one tool, `train_test_split`, which **randomly deals a pile of data into two smaller piles**.
- First line: split everything into **85% "temporary"** and **15% "test."** The test pile is now locked in a drawer.
- Second line: split that temporary pile again into **train** and **validation**. (`0.1765` of 85% works out to 15% of the original — giving a final **70 / 15 / 15** split.)
- The `x_tmp, x_test = ...` style means "this function hands back *two* things at once; put them in these two boxes."

**Why three piles?**
- **Train** — the network studies this and adjusts itself to fit it. (Homework.)
- **Validation** — we peek at this *between* experiments to tune our choices and decide when to stop. (A practice exam.)
- **Test** — opened **once**, at the very end, for an honest final grade. (The real exam.)

The golden rule the code is enforcing: **the network must never study the test pile.** If it did, a good score would just mean it memorized the exam. The rest of the cell draws the three piles in three colors.

---

## Part 4 — Turning numbers into "tensors"

```python
def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32).view(-1, 1)

X_train, Y_train = to_tensor(x_train), to_tensor(y_train)
...
print("X_train shape:", tuple(X_train.shape), " Y_train shape:", tuple(Y_train.shape))
```

**In plain words:**
- PyTorch doesn't work with plain lists of numbers; it works with **tensors** (its own kind of number-grid). This little machine `to_tensor` converts our numbers into that format.
- `dtype=torch.float32` — "store them as decimals" (the standard for neural nets).
- `.view(-1, 1)` — **reshape** the numbers into a tall single column: one row per example, one column (because each input is just one number). The `-1` means "you figure out how many rows; I just want 1 column."
- We convert all six piles (the `x` and `y` of train, val, test).
- The `print` shows the **shape** — e.g. `(168, 1)` = *168 examples, 1 feature each*. Checking shapes is how you sanity-check that data is laid out right.

**Why:** it's just plugging our data into the shape PyTorch expects — like decanting ingredients into the specific bowls the recipe needs.

---

## Part 5 — 🧠 THE NEURAL NETWORK (the most important part)

This is the brain. Read this part slowly.

```python
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=16, out_dim=1, depth=2, p_drop=0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            if p_drop > 0:
                layers += [nn.Dropout(p_drop)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### First, what *is* a neural network?

Picture a row of light-switches and dimmer knobs wired together. A number goes in one end; at each stage it gets multiplied, added, and gently bent; a number comes out the other end. **The knobs are the "weights."** *Learning* means slowly turning all the knobs until the output matches what we wanted. That's the entire idea. Everything else is detail.

### Line by line

- `class MLP(nn.Module):` — a **class** is a *blueprint* for building a thing (here, a network). `MLP` = **M**ulti-**L**ayer **P**erceptron, the classic simple network. `(nn.Module)` means "build on top of PyTorch's official network blueprint, so we inherit all its machinery for free" (like how a new car design still inherits wheels and an engine).

- `def __init__(self, ...):` — the **`__init__`** is the **"assembly instructions"**: what to build the moment we create a network. Its options (with sensible defaults) are:
  - `in_dim=1` — how many numbers go in (1, because our input `x` is a single number).
  - `hidden=16` — how many neurons in each hidden layer (its "width," its brain-power).
  - `out_dim=1` — how many numbers come out (1, our predicted `y`).
  - `depth=2` — how many hidden layers to stack (its "depth").
  - `p_drop=0.0` — how much dropout to use (0 = none, for now).

- `super().__init__()` — "run PyTorch's own setup first." A required formality that switches on the inherited machinery.

- `layers, d = [], in_dim` — start an **empty list** called `layers` (we'll fill it with the network's stages), and a tracker `d` remembering "how many numbers are flowing right now" (starts at the input size).

- `for _ in range(depth):` — **"repeat the following `depth` times."** (The `_` just means "we don't care about the counter, only that it repeats.") So with `depth=2`, we build two hidden stages.

  - `layers += [nn.Linear(d, hidden), nn.Tanh()]` — add **two** pieces each time:
    - **`nn.Linear(d, hidden)`** — a **Linear layer**: it takes the `d` incoming numbers, and produces `hidden` new numbers, where **each new number is a weighted sum of the incoming ones plus a bias**. *These weights and biases are the knobs that get learned.* This is where the network's actual "memory" lives.
    - **`nn.Tanh()`** — the **activation**, the *bend*. It takes each number and squashes it smoothly into the range −1 to 1. **Why it's essential:** without a bend between linear layers, stacking them just gives one big straight line — and a straight line can never trace our wavy curve. The bend is what lets the network *curve*.
  - `if p_drop > 0: layers += [nn.Dropout(p_drop)]` — *only if* we asked for dropout, add a **Dropout** stage (randomly switches off some neurons during training, for robustness). With the default `0.0`, this is skipped.
  - `d = hidden` — update the tracker: from now on, `hidden` numbers are flowing.

- `layers += [nn.Linear(d, out_dim)]` — after the hidden stages, add **one final Linear layer** that collapses everything down to `out_dim` = **1 number**: the prediction. (No bend on the end — we want the raw number.)

- `self.net = nn.Sequential(*layers)` — **`Sequential`** is a "pipeline" that chains all those stages in order, so a number flows through them one after another. `self.net` = "store this pipeline inside me as *my* network." (`self` = "this particular network"; the `*` just means "unpack the list into the pipeline.")

- `def forward(self, x):` / `return self.net(x)` — **`forward`** defines **what happens when you make a prediction**: take input `x`, push it through the pipeline, return whatever comes out. This *is* the **forward pass**.

### So what did we actually build?

With the defaults, the pipeline is:

```
input (1 number)
  → Linear: 1 → 16     (16 weighted sums)
  → Tanh   (bend)
  → Linear: 16 → 16
  → Tanh   (bend)
  → Linear: 16 → 1     (squash down to the prediction)
output (1 number)
```

That's the **1 → 16 → 16 → 1** shape.

```python
model = MLP(hidden=16, depth=2)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\nTotal adjustable weights: {n_params}")
```

- `model = MLP(...)` — **actually build one** network from the blueprint.
- `n_params = sum(p.numel() for p in model.parameters())` — count **every knob** in the network. In plain words: "go through all the network's adjustable numbers and add up how many there are." That count is how much "capacity" the model has to learn.
- The prints show the network's structure and the total knob count.

**One-sentence explanation you can give anyone:** *"It's a little machine made of stages; each stage multiplies the numbers by adjustable knobs and gently bends them; learning is just tuning all those knobs until the machine's output matches reality."*

---

## Part 6 — Watching one prediction flow through (the forward pass)

```python
sample = X_train[:5]
print(f"{'input':16s} -> {tuple(sample.shape)}")
h = sample
for layer in model.net:
    h = layer(h)
    print(f"{layer.__class__.__name__:16s} -> {tuple(h.shape)}")
```

**In plain words:**
- `sample = X_train[:5]` — take the **first 5** training inputs to experiment with.
- `h = sample` — put them in a box `h` (our "current numbers as they travel").
- `for layer in model.net:` — **go through each stage of the pipeline, one at a time.**
  - `h = layer(h)` — push the current numbers through this stage; the result becomes the new `h`.
  - the `print` shows the **shape** after each stage.

**Why:** it makes the invisible *visible*. You literally watch 5 inputs of shape `(5,1)` enter, widen to `(5,16)` in the hidden layers, and shrink back to `(5,1)` — 5 predictions. It proves the network processes a whole batch at once and shows the "shape story" of a forward pass.

---

## Part 7 — Weight initialization (the starting knobs)

```python
def apply_init(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "zeros":
                nn.init.zeros_(m.weight)
            elif scheme == "too big":
                nn.init.normal_(m.weight, std=3.0)
            elif scheme == "xavier (good)":
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model
```

**In plain words:**
- Before training, every knob has to start *somewhere*. This machine sets those starting values three different ways so we can compare them.
- `for m in model.modules():` — go through every piece of the network.
- `if isinstance(m, nn.Linear):` — "if this piece is a Linear layer (i.e. has knobs)…"
- then, depending on the chosen `scheme`, set its weights to: **all zeros**, **random but too large** (`std=3.0` = big spread), or **Xavier** (a smart, well-scaled random start).
- `nn.init.zeros_(m.bias)` — set the biases to zero.

**Why:** the rest of the cell draws what an *untrained* network outputs for each start. You see **zeros = a dead flat line** (all neurons identical, stuck), **too big = a wild jagged shape**, and **Xavier = a gentle wiggle ready to be shaped**. The lesson: *where you start affects whether learning can even begin.*

---

## Part 8 — Measuring wrongness (the loss / MSE)

```python
loss_fn = nn.MSELoss()
pred = model(X_train)
mse_by_hand = ((pred - Y_train) ** 2).mean().item()
mse_pytorch = loss_fn(pred, Y_train).item()
```

**In plain words:**
- `loss_fn = nn.MSELoss()` — grab PyTorch's ready-made **Mean Squared Error** measuring stick.
- `pred = model(X_train)` — ask the (untrained) network to predict on all training inputs. *(Writing `model(...)` automatically runs the `forward` we defined.)*
- `mse_by_hand = ((pred - Y_train) ** 2).mean()` — compute the error **by hand** to demystify it: take each *(prediction − truth)*, **square it** (`** 2`), then take the **average** (`.mean()`). `.item()` just pulls the single number out of its tensor wrapper.
- `mse_pytorch = loss_fn(...)` — the same thing using PyTorch's tool.
- They print **identical** — proving MSE is nothing mysterious: *"average of the squared misses."*

**Why square the misses?** So overshooting and undershooting both count as positive, and **big mistakes hurt much more than small ones** — which pushes the network to avoid large errors. This one number is the network's "how wrong am I?" score, and the entire goal of training is to make it small.

---

## Part 9 — "Nudge one knob, watch the loss move" (the key idea)

```python
layer0 = model.net[0]
i, j = 0, 0
w0 = layer0.weight.data[i, j].item()

sweep = np.linspace(w0 - 4, w0 + 4, 160)
losses = []
with torch.no_grad():
    for val in sweep:
        layer0.weight.data[i, j] = val
        losses.append(loss_fn(model(X_train), Y_train).item())
    layer0.weight.data[i, j] = w0

model.zero_grad()
loss_here = loss_fn(model(X_train), Y_train)
loss_here.backward()
grad = layer0.weight.grad[i, j].item()
```

**In plain words:**
- `layer0 = model.net[0]` — grab the first Linear layer. `w0 = ...weight.data[0,0]` — read the value of **one single knob** in it, and remember it.
- `sweep = np.linspace(w0 - 4, w0 + 4, 160)` — make a list of 160 values to *try* for that one knob, ranging from a bit below to a bit above its current value.
- `with torch.no_grad():` — "we're just looking, not learning here" (turns off the learning machinery to be fast).
  - the loop sets that one knob to each trial value and records the resulting **loss** each time. Every other knob stays frozen.
  - the last line **puts the knob back** to its original value (we were only experimenting).
- Then the last three lines compute the **gradient** at the real value:
  - `model.zero_grad()` — clear any old slope info.
  - `loss_here = loss_fn(...)` then `loss_here.backward()` — **`.backward()` is the magic word**: it asks PyTorch to work out the **slope of the loss** with respect to every knob (this is *backpropagation*, done automatically).
  - `grad = layer0.weight.grad[i,j]` — read out the slope for our one knob.

**Why this is the heart of everything:** the recorded losses, plotted, form a **curve** — moving that one knob left or right changes how wrong the model is. The **gradient is just the steepness of that curve** where we're standing. It's an arrow saying *"to reduce the loss, move the knob this way."* The chart even draws a dashed tangent line whose slope equals the gradient — showing they're the same thing. **Training is doing exactly this for all knobs at once, over and over.**

---

## Part 10 — Rolling downhill (the optimizer step)

```python
def roll_downhill(lr, n_steps=14, start_offset=3.5):
    layer0.weight.data[i, j] = w0 + start_offset
    path_w, path_loss = [], []
    for _ in range(n_steps):
        model.zero_grad()
        loss = loss_fn(model(X_train), Y_train)
        loss.backward()
        g = layer0.weight.grad[i, j].item()
        cur = layer0.weight.data[i, j].item()
        path_w.append(cur); path_loss.append(loss.item())
        layer0.weight.data[i, j] = cur - lr * g
    layer0.weight.data[i, j] = w0
    return path_w, path_loss
```

**In plain words:** this machine takes one knob, starts it off to the side (`w0 + 3.5`), and repeatedly steps it **downhill** on the loss curve:
- Each loop: clear old slopes → measure the loss → `.backward()` to get the slope `g` → read the knob's current value `cur` → record where we are → then the crucial line:
- **`layer0.weight.data[i, j] = cur - lr * g`** — *this is gradient descent, written out by hand.* New value = old value − (step size × slope). Subtracting the slope moves us downhill; `lr` (**learning rate**) controls how big each step is.
- Afterwards it restores the knob and hands back the path, so we can draw the "ball rolling to the bottom."

**Why:** it makes the abstract update rule tangible — you *see* a knob start high on the curve and hop, step by step, into the valley. The next cell reruns this with a **too-small**, **just-right**, and **too-big** learning rate, showing that step size is the make-or-break setting: too small crawls, too big overshoots and bounces. In the real training, an **optimizer** (Adam) does this same `− lr × slope` update to *every* knob automatically.

---

## Part 11 — Batch, iteration, epoch

```python
from torch.utils.data import TensorDataset, DataLoader
batch_size = 16
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
iters_per_epoch = len(train_loader)
```

**In plain words:**
- `TensorDataset(X_train, Y_train)` — pair up each input with its correct answer into one dataset.
- `DataLoader(..., batch_size=16, shuffle=True)` — a **conveyor belt** that serves the data in **small handfuls of 16** (`batch_size`), in a **shuffled** order each pass.
- `iters_per_epoch = len(train_loader)` — how many handfuls it takes to get through everything.

**The three words, made concrete:**
- **Batch** = one handful (16 examples).
- **Iteration** = processing one handful and updating the knobs once.
- **Epoch** = one full pass over *all* the data (here, 168 ÷ 16 ≈ **11 iterations**).

**Why handfuls?** Updating after every small batch is faster and steadier than waiting to see all the data each time, and the little bit of shuffle-noise actually helps learning. The rest of the cell prints the batches to prove there are 11 per epoch.

---

## Part 12 — The training loop (where learning actually happens)

```python
def train_model(model, train_loader, epochs=250, lr=0.01, weight_decay=0.0, snapshot_epochs=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    ...
    for ep in range(epochs + 1):
        model.eval()
        with torch.no_grad():
            history["train"].append(loss_fn(model(X_train), Y_train).item())
            history["val"].append(loss_fn(model(X_val), Y_val).item())
            ...
        if ep == epochs:
            break
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
```

This is the engine. **In plain words:**
- `optimizer = torch.optim.Adam(model.parameters(), lr=..., weight_decay=...)` — hire an **optimizer** named **Adam** and hand it **all the knobs** (`model.parameters()`). Its job: nudge every knob downhill each step. `weight_decay` is a gentle "keep the knobs small" pressure (regularization, off by default).
- `history = {"train": [], "val": []}` — two empty notebooks to record the loss after each epoch.
- `for ep in range(epochs + 1):` — repeat for each epoch.
  - `model.eval()` + `with torch.no_grad():` — "just measuring now, not learning": record the current train and validation loss into `history`. (Doing this *before* training in each epoch means epoch 0 is the untrained score.)
  - `if ep == epochs: break` — after the final measurement, stop.
  - `model.train()` — switch into learning mode.
  - `for xb, yb in train_loader:` — go through the conveyor belt, one handful at a time. `xb` = the batch of inputs, `yb` = their correct answers. **These four lines are the whole heartbeat of deep learning:**
    1. `optimizer.zero_grad()` — **wipe the previous slopes** (they'd otherwise pile up).
    2. `loss = loss_fn(model(xb), yb)` — **forward pass** + measure how wrong we are on this handful.
    3. `loss.backward()` — **backpropagation**: compute the slope for every knob.
    4. `optimizer.step()` — **nudge every knob downhill** by (learning rate × its slope).

Repeat that millions of times and the network gets good. **Those four lines are the single most important thing in the whole notebook** — memorize them: *clear, forward+measure, backward, step.*

**Why record two losses?** Watching **train** vs **validation** loss is how we later spot overfitting. The next cell plots the losses falling and shows snapshots of the fitted curve at epochs 0, 5, 20, 60, 250 — you literally watch a flat line bend into the true curve.

---

## Part 13 — Causing (then curing) overfitting

```python
small_idx = np.arange(25)
Xs_train, Ys_train = X_train[small_idx], Y_train[small_idx]
...
def fit_variant(weight_decay=0.0, p_drop=0.0, epochs=400):
    m = MLP(hidden=128, depth=3, p_drop=p_drop)
    ...
h_over, c_over, gx = fit_variant()
h_wd,   c_wd,   _  = fit_variant(weight_decay=1e-2)
h_do,   c_do,   _  = fit_variant(p_drop=0.25)
```

**In plain words:**
- `Xs_train = X_train[:25]` — deliberately use only **25** training points — very little data, easy to memorize.
- `fit_variant(...)` builds a **deliberately oversized** network (`hidden=128, depth=3` — way more knobs than needed) and trains it. Its options let us turn on cures:
  - `fit_variant()` — no cure → it **overfits** (memorizes the 25 noisy dots).
  - `fit_variant(weight_decay=1e-2)` — turn on **L2 regularization** (the "keep knobs small" pressure).
  - `fit_variant(p_drop=0.25)` — turn on **dropout** (randomly switch off 25% of neurons while training).

**Why:** the next cells draw all three side by side. The no-cure model makes a wild, jagged line threading every noisy dot — and its **validation** loss climbs even as its **train** loss keeps dropping (the classic overfitting fingerprint). The regularized and dropout models stay **smooth** and follow the true curve. This is the whole lesson of generalization in one picture: *a model that memorizes the practice data fails on new data.*

---

## Part 14 — Hyperparameter tuning

```python
def quick_train(hidden, lr, weight_decay, epochs=160):
    m = MLP(hidden=hidden, depth=2)
    ...
    return loss_fn(m(X_val), Y_val).item()

grid = {"lr": [0.001, 0.01, 0.05], "hidden": [8, 32], "weight_decay": [0.0, 1e-3]}
rows = []
for lr, hidden, wd in itertools.product(grid["lr"], grid["hidden"], grid["weight_decay"]):
    rows.append({"lr": lr, "hidden": hidden, "weight_decay": wd, "val_loss": quick_train(hidden, lr, wd)})
```

**In plain words:**
- `quick_train(...)` — a mini machine: build a network with given settings, train it, and **return its validation loss** (its report card).
- `grid = {...}` — the **menu of settings to try**: 3 learning rates × 2 network sizes × 2 regularization strengths.
- `itertools.product(...)` — "make **every possible combination**" of that menu (3 × 2 × 2 = **12** recipes).
- the `for` loop trains all 12 and records each one's validation loss in `rows`.

**Why:** some settings (the **hyperparameters**) can't be learned by gradient — *we* have to choose them. So we try many, **rank them by validation score** (never test!), and pick the winner. The next cells draw a ranked bar chart, then retrain the best recipe and — for the **first and only time** — check the **test** set for an honest final grade (test MSE ≈ **0.0755**, essentially as good as possible given the noise).

---

## Part 15 — The real-data capstone (California housing)

```python
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing()
    ...
scaler = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
```

**In plain words:**
- `try: ... except: ...` — "attempt to download the California housing data; if there's no internet, fall back to a built-in dataset instead" (so the notebook always runs).
- This is **real** data: 20,640 districts, 8 facts each (income, house age, rooms, location…), predicting the median house value.
- `StandardScaler().fit(Xtr)` then `.transform(...)` — **feature scaling**: real features live on wildly different scales (incomes in the tens of thousands vs. latitude around 37). We rescale them all to a comparable range so no single feature bullies the others. **Crucially, we `fit` the scaler on the training data only**, then apply it to val/test — so no information leaks from the test set.

```python
best_val, best_state, patience, wait = float("inf"), None, 20, 0
for ep in range(300):
    ...
    if va < best_val - 1e-4:
        best_val, best_state, wait = va, {...copy of weights...}, 0
    else:
        wait += 1
        if wait >= patience:
            break
cap_model.load_state_dict(best_state)
```

**In plain words — this is "early stopping":**
- Keep a record of the **best validation score so far** (`best_val`) and a **saved copy of the knobs** at that best moment (`best_state`).
- Each epoch: if validation improved, save this as the new best and reset the patience counter. If it *didn't* improve, add 1 to `wait`. Once we've gone `patience=20` epochs with **no improvement**, **stop early** — more training would just start overfitting.
- `load_state_dict(best_state)` — at the end, **restore the knobs from the best epoch**, not the last one.

**Why:** it's an automatic "quit while you're ahead." The final cell then scores the untouched test set and reports **RMSE ≈ 0.512** and **R²** — proving the same recipe that learned a toy curve also works on messy real data.

---

## The 60-second version you can say out loud

> *"I built a small artificial brain — a network of stages full of adjustable knobs. I made fake data from a hidden curve plus random noise, and split it into three piles: one to learn from, one to check myself against, and one sealed for a final test. To teach the network, I measured how wrong it was with a single score (squared error), then used calculus to find, for every knob, which way to turn it to lower that score, and nudged them all a tiny bit in that direction. Repeat that thousands of times and the network rediscovers the hidden curve. I also showed how it can 'cheat' by memorizing (overfitting) and two ways to stop that, how to pick the best settings by testing on the validation pile, and finally proved it all works on real California house-price data."*

Every claim in that paragraph maps to a part above. If you can say that and point at the pictures, you understand this project better than most people who use these tools daily.
