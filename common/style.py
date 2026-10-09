"""One look for every figure in the series.

White background, Space Grotesk (bundled in common/fonts, OFL), and the palette
of athithyaloganathan.com used sparingly:

  TEAL   the thing the sentence in the post is about (the new method, the fix)
  CLAY   the problem or the "before" (the baseline that loses, the miscalibration)
  greys  everything that is context

Lines are labelled directly where a legend can be avoided; no top/right spines,
no grids unless a value has to be read off.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK = "#18313a"       # text and axes: the site's deep ink
TEAL = "#126a72"      # accent 1
CLAY = "#a34f3e"      # accent 2
GREY = "#8f9a9d"      # context series
LIGHT = "#dfe3e3"     # fills, reference lines
FAINT = "#f1f3f3"     # "nothing to see here" areas
WHITE = "#ffffff"

ACCENT = TEAL
# Names older scripts import.
PAPER, LEAF, MUTED, GRID = WHITE, TEAL, GREY, LIGHT
SERIES = [TEAL, CLAY, GREY, INK]

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FAMILY = "DejaVu Sans"
for _f in sorted(FONT_DIR.glob("SpaceGrotesk-*.ttf")):
    font_manager.fontManager.addfont(str(_f))
    FAMILY = "Space Grotesk"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "font.family": [FAMILY, "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "font.weight": "regular",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.labelsize": 10.5,
        "axes.titlecolor": INK,
        "axes.titlesize": 13.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path}")
