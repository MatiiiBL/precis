"""Figures for the benchmark results (PNG, for the docs and the README).

Matplotlib is a development dependency: the figures are made when the
benchmarks are run from a checkout, not by an installed ``precis``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Categorical slots and neutral inks of the reference palette (light mode).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class Series:
    label: str
    values: Sequence[float]


@dataclass(frozen=True)
class Panel:
    title: str
    unit: str
    series: Sequence[Series]
    reference: float | None = None
    """A horizontal reference line (a limit), labelled on the right."""
    reference_label: str = ""


def _style(ax: Axes, panel: Panel, x: Sequence[float], right_margin: float) -> None:
    ax.set_facecolor(SURFACE)
    several = len(panel.series) > 1
    ax.set_title(panel.title, loc="left", fontsize=11, color=INK, pad=24 if several else 8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(x), [str(int(v)) for v in x])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.25, x[-1] * right_margin)
    ax.set_xlabel("concurrency ceiling", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(panel.unit, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    top = max(max(s.values, default=0.0) for s in panel.series)
    if panel.reference is not None:
        top = max(top, panel.reference)
    ax.set_ylim(0, top * 1.12 if top > 0 else 1.0)


def small_multiples(x: Sequence[float], panels: Sequence[Panel], path: Path, *, title: str) -> Path:
    """Draw ``panels`` side by side over the same x values and save a PNG.

    Every panel has its own single y axis; series in a panel share it and
    are labelled at their last point as well as in a legend.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("figures need matplotlib: uv sync --group dev") from exc

    # The same x range in every panel, with room on the right for the labels
    # at the last point of multi-series panels.
    right_margin = 2.2 if any(len(p.series) > 1 for p in panels) else 1.25
    columns = min(len(panels), 2)
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.4 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, panel in zip(axes.flat, panels, strict=False):
        _style(ax, panel, x, right_margin)
        for color, series in zip(SERIES, panel.series, strict=False):
            ax.plot(x, series.values, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=series.label)  # fmt: skip
            if len(panel.series) > 1:
                ax.annotate(series.label, (x[-1], series.values[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8,
                            color=INK_SECONDARY)  # fmt: skip
        if panel.reference is not None:
            ax.axhline(panel.reference, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(panel.reference_label, (x[0], panel.reference), xytext=(0, -11),
                        textcoords="offset points", fontsize=8, color=INK_SECONDARY)  # fmt: skip
        if len(panel.series) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0),
                      ncol=len(panel.series), labelcolor=INK_SECONDARY, borderaxespad=0.2,
                      handlelength=1.6)  # fmt: skip
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
