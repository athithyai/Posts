"""One look for every figure in the series, taken from the website's palette."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#18313a"
PAPER = "#f6f5ef"
TEAL = "#126a72"
LEAF = "#47744c"
CLAY = "#a34f3e"
MUTED = "#8a9a9e"
GRID = "#dcdcd2"

SERIES = [TEAL, CLAY, LEAF, INK, MUTED]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
