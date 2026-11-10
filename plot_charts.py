# Draws the loss curve and the accuracy curve from training_log.csv
# Run from the project folder:  python3 plot_charts.py
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- read the log (plain loops) ----
epochs = []
train_loss = []
train_acc = []
val_acc = []

with open("training_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]))
        val_acc.append(float(row["val_acc"]))

# ---- colours (blue = training, orange = validation) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)

last = epochs[-1]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel("Epoch", fontsize=9.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(0.5, max(last, 20) + 4.5)
    ax.set_xticks([1, 5, 10, 15, 20])


# ---- chart 1: loss ----
style(ax1, "Training loss", "Loss")
guess_loss = math.log(20)
ax1.axhline(guess_loss, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax1.text(max(last, 20) + 4.3, guess_loss + 0.04, "guessing 2.9957", ha="right", fontsize=8.5, color=MUTED)
ax1.plot(epochs, train_loss, color=BLUE, linewidth=2, marker="o", markersize=6,
         markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax1.set_ylim(0, 3.3)
ax1.text(last + 0.5, train_loss[-1], "%.2f" % train_loss[-1], va="center", fontsize=9.5, color=INK)

# ---- chart 2: accuracy ----
style(ax2, "Accuracy (0 to 1)", "Accuracy")
ax2.axhline(0.05, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax2.text(max(last, 20) + 4.3, 0.065, "guessing 0.05", ha="right", fontsize=8.5, color=MUTED)
ax2.plot(epochs, train_acc, color=BLUE, linewidth=2, marker="o", markersize=6,
         markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Training")
ax2.plot(epochs, val_acc, color=ORANGE, linewidth=2, marker="o", markersize=6,
         markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Validation")
ax2.set_ylim(0, 1.0)

# end labels: the higher line's label goes up, the lower one goes down, so they never overlap
if val_acc[-1] >= train_acc[-1]:
    val_shift = 0.025
    train_shift = -0.025
else:
    val_shift = -0.025
    train_shift = 0.025
ax2.text(last + 0.5, val_acc[-1] + val_shift, "Validation %.4f" % val_acc[-1], va="center", fontsize=9.5, color=INK)
ax2.text(last + 0.5, train_acc[-1] + train_shift, "Training %.4f" % train_acc[-1], va="center", fontsize=9.5, color=INK)
ax2.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)

fig.tight_layout(pad=1.6)
fig.savefig("training_charts.png", facecolor=SURFACE)
print("saved training_charts.png with", len(epochs), "epochs")
