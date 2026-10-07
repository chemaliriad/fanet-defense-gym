"""Matplotlib rendering of a simulator snapshot (privileged view, for humans only).

Colours follow a validated categorical palette (slots 1-3 pass colour-vision-deficiency
checks in light and dark mode); node state is also encoded by marker shape, so colour is
never the only channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scenario import ScenarioSpec

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "s1": "#2a78d6",  # benign
        "s2": "#eb6834",  # attacking
        "s3": "#1baf7a",  # dormant implant
        "s4": "#eda100",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
        "s4": "#c98500",
    },
}
STATES = (  # (label, palette slot, marker)
    ("benign", "s1", "o"),
    ("attacking", "s2", "D"),
    ("dormant implant", "s3", "^"),
)


def node_states(snap: dict[str, Any]) -> list[str]:
    n = snap["gcs"]
    out = []
    for i in range(n):
        if snap["offline"][i]:
            out.append("offline")
        elif snap["misbehaving"][i]:
            out.append("attacking")
        elif snap["compromised"][i]:
            out.append("dormant implant")
        else:
            out.append("benign")
    return out


def draw_snapshot(
    ax: Any,
    snap: dict[str, Any],
    spec: ScenarioSpec,
    theme: str = "light",
    title: str | None = None,
) -> None:
    th = THEMES[theme]
    pos, gcs = snap["pos"], snap["gcs"]
    n = gcs
    usable, blocked, nh = snap["usable"], snap["blocked"], snap["next_hop"]
    ax.set_facecolor(th["surface"])
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if blocked[i, j]:
                ax.plot(*pos[[i, j]].T, ls=(0, (4, 3)), lw=1.2, color=th["ink2"], zorder=1)
            elif usable[i, j]:
                ax.plot(*pos[[i, j]].T, lw=1.0, color=th["grid"], zorder=1)
    for i in range(n):
        if nh[i] >= 0:
            ax.annotate(
                "",
                xy=pos[nh[i]],
                xytext=pos[i],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": th["muted"],
                    "lw": 0.8,
                    "shrinkA": 6,
                    "shrinkB": 6,
                },
                zorder=2,
            )
    states = node_states(snap)
    for label, slot, marker in STATES:
        idx = [i for i, s in enumerate(states) if s == label]
        if idx:
            ax.scatter(
                *pos[idx].T,
                s=64,
                marker=marker,
                color=th[slot],
                edgecolor=th["surface"],
                linewidth=2,
                zorder=3,
            )
    off = [i for i, s in enumerate(states) if s == "offline"]
    if off:
        ax.scatter(
            *pos[off].T,
            s=64,
            marker="o",
            facecolor="none",
            edgecolor=th["muted"],
            linewidth=1.5,
            zorder=3,
        )
    ax.scatter(
        *pos[gcs],
        s=120,
        marker="s",
        color=th["ink"],
        edgecolor=th["surface"],
        linewidth=2,
        zorder=4,
    )
    ax.set_xlim(0, spec.arena)
    ax.set_ylim(0, spec.arena)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(th["grid"])
    ax.set_title(title or f"t = {snap['t']} s", fontsize=10, color=th["ink"], loc="left")


def legend_handles(theme: str = "light") -> list[Any]:
    from matplotlib.lines import Line2D

    th = THEMES[theme]
    handles = [
        Line2D([], [], marker=m, ls="", color=th[slot], markersize=7, label=label)
        for label, slot, m in STATES
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor=th["muted"],
            markersize=7,
            label="offline (re-flashing)",
        ),
        Line2D([], [], marker="s", ls="", color=th["ink"], markersize=7, label="ground station"),
        Line2D([], [], ls=(0, (4, 3)), color=th["ink2"], label="blocked link"),
        Line2D([], [], color=th["muted"], marker=">", markersize=4, label="route to station"),
    ]
    return handles


def render_frame(
    snap: dict[str, Any], spec: ScenarioSpec, title: str | None = None, theme: str = "light"
) -> np.ndarray:
    """RGB array of one snapshot (used by ``env.render()`` and the ``demo`` command)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110, facecolor=th["surface"])
    draw_snapshot(ax, snap, spec, theme, title)
    leg = ax.legend(
        handles=legend_handles(theme),
        loc="upper right",
        fontsize=7,
        facecolor=th["surface"],
        edgecolor=th["grid"],
        labelcolor=th["ink2"],
    )
    leg.get_frame().set_alpha(0.92)
    fig.tight_layout()
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
