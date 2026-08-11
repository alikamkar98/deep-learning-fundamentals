"""
viz.py — small, purely-cosmetic plotting helpers for the notebook.

Nothing here contains machine-learning logic. It only keeps the *look* of the
plots consistent (one shared palette, one style) and provides a reusable routine
for drawing a neural-network architecture diagram. All the actual learning code
lives in the notebook so it stays visible.
"""

import matplotlib.pyplot as plt
import numpy as np

# One shared palette so every plot in the notebook speaks the same visual language.
COLORS = {
    "true":  "#1f3a5f",   # the underlying function we're trying to recover (navy)
    "train": "#2563eb",   # training data (blue)
    "val":   "#f59e0b",   # validation data (amber)
    "test":  "#e11d48",   # test data (rose)
    "pred":  "#10b981",   # model predictions (green)
    "grid":  "#d7dce3",
    "ink":   "#1f2937",
}


def set_style():
    """Apply a clean, consistent matplotlib style. Call once at the top."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#c3cad6",
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.8,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.labelcolor":   COLORS["ink"],
        "xtick.color":       COLORS["ink"],
        "ytick.color":       COLORS["ink"],
        "text.color":        COLORS["ink"],
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "figure.dpi":        110,
        "savefig.dpi":       110,
    })


def draw_mlp(ax, layer_sizes, title="Network architecture", input_labels=None):
    """
    Draw a fully-connected multi-layer perceptron as circles-and-lines.

    layer_sizes : list[int]  e.g. [1, 8, 8, 1]  → 1 input, two hidden layers of 8, 1 output.
    This is a *picture only*; it does not reflect trained weights.
    """
    ax.axis("off")
    ax.set_title(title)
    n_layers = len(layer_sizes)
    v_gap = 1.0
    h_gap = 2.2
    # store node positions per layer
    positions = []
    max_nodes = max(layer_sizes)
    for li, n in enumerate(layer_sizes):
        xs = li * h_gap
        # vertically center each layer
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * v_gap
        positions.append([(xs, y) for y in ys])

    # edges first (so nodes draw on top)
    for li in range(n_layers - 1):
        for (x0, y0) in positions[li]:
            for (x1, y1) in positions[li + 1]:
                ax.plot([x0, x1], [y0, y1], color="#c3cad6", linewidth=0.6, zorder=1)

    # nodes
    layer_colors = [COLORS["train"]] + ["#7c93b8"] * (n_layers - 2) + [COLORS["pred"]]
    layer_names = ["input"] + [f"hidden {i+1}" for i in range(n_layers - 2)] + ["output"]
    for li, layer in enumerate(positions):
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), 0.18, color=layer_colors[li],
                                    ec="white", lw=1.5, zorder=2))
        # layer label under the column
        ax.text(li * h_gap, -(max_nodes / 2) - 0.6, layer_names[li],
                ha="center", va="top", fontsize=9, color=COLORS["ink"])
        ax.text(li * h_gap, (max_nodes / 2) + 0.5, str(layer_sizes[li]),
                ha="center", va="bottom", fontsize=10, weight="bold",
                color=COLORS["ink"])

    ax.set_xlim(-0.8, (n_layers - 1) * h_gap + 0.8)
    ax.set_ylim(-(max_nodes / 2) - 1.2, (max_nodes / 2) + 1.2)
    ax.set_aspect("equal")
