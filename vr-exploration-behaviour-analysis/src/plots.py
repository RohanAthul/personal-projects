"""Figures. Colours: two validated categorical slots (3DGS baseline, ZipNeRF) and a one-hue blue
sequential ramp for dwell maps; neutral ink for all text."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

GROUP_COLORS = {"3DGS": "#2a78d6", "ZipNeRF": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

LABELS = {
    "ang_speed_deg_s": "Angular speed (deg/s, median)",
    "trans_speed_m_s": "Translation speed (m/s, median)",
    "dir_entropy_bits": "Viewing-direction entropy (bits)",
    "dir_coverage": "Viewing-sphere coverage (share of bins)",
    "pos_entropy_bits": "Floor-position entropy (bits)",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2, "savefig.dpi": 150,
    })


def dwell_maps(dwell: dict, scenes: list[str], groups: dict, n_yaw: int, n_pitch: int):
    """Per-scene share of time per equal-area direction bin, pooled over users."""
    fig, axes = plt.subplots(3, 4, figsize=(11, 5.6), sharex=True, sharey=True, constrained_layout=True)
    maps = {}
    for sc in scenes:
        c = sum(v for (u, s), v in dwell.items() if s == sc).astype(float)
        maps[sc] = 100 * c.reshape(n_pitch, n_yaw) / c.sum()
    vmax = np.percentile(np.concatenate([m.ravel() for m in maps.values()]), 99.5)
    sin_edges = np.linspace(-1, 1, n_pitch + 1)
    for ax, sc in zip(axes.ravel(), scenes):
        im = ax.pcolormesh(np.linspace(-180, 180, n_yaw + 1), sin_edges, maps[sc], cmap=BLUES,
                           vmin=0, vmax=vmax, shading="flat")
        ax.set_title(f"{sc} ({groups[sc]})", color=INK)
        ax.grid(False)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks(np.sin(np.radians([-60, -30, 0, 30, 60])), ["-60", "-30", "0", "30", "60"])
    for ax in axes[-1]:
        ax.set_xlabel("Yaw (deg, scene frame)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pitch (deg)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Share of viewing time per bin (%)")
    fig.suptitle("Where users looked: dwell time per equal-area direction bin (first 60 s, all users)")
    return fig


def time_course(windows: pd.DataFrame, measures: list[str]):
    """Mean across users (per-user mean over scenes in the group) +/- 95% t-CI per time window."""
    from scipy import stats
    w = windows.dropna(subset=["t_mid_s"])
    per_user = w.groupby(["group", "user", "t_mid_s"])[measures].mean().reset_index()
    fig, axes = plt.subplots(1, len(measures), figsize=(3.6 * len(measures), 3.2), constrained_layout=True)
    for ax, m in zip(axes, measures):
        for g, col in GROUP_COLORS.items():
            d = per_user[per_user.group == g].groupby("t_mid_s")[m]
            mean, sem, n = d.mean(), d.sem(), d.count()
            half = sem * stats.t.ppf(0.975, n - 1)
            ax.fill_between(mean.index, mean - half, mean + half, color=col, alpha=0.18, linewidth=0)
            ax.plot(mean.index, mean, color=col, label=g)
        ax.set_xlabel("Time from scene start (s, centre of 10 s window)")
        ax.set_title(LABELS[m])
    axes[0].legend(frameon=False, title="Scene group")
    fig.suptitle("Exploration over time: mean across users, shaded 95% CI")
    return fig


def paired_panels(wide: dict, a: str, b: str, title: str, colors: tuple[str, str]):
    """One panel per measure: each user's value under condition a and b, joined by a line."""
    fig, axes = plt.subplots(1, len(wide), figsize=(2.5 * len(wide), 3.4), constrained_layout=True)
    for ax, (m, df) in zip(axes, wide.items()):
        for _, r in df.iterrows():
            ax.plot([0, 1], [r[a], r[b]], color=GRID, linewidth=1, zorder=1)
        for x, cond, col in ((0, a, colors[0]), (1, b, colors[1])):
            ax.scatter(np.full(len(df), x), df[cond], s=22, color=col, edgecolor=SURFACE, linewidth=1, zorder=2)
            ax.hlines(df[cond].median(), x - 0.18, x + 0.18, color=INK, linewidth=2, zorder=3)
        ax.set_xticks([0, 1], [a, b])
        ax.set_xlim(-0.4, 1.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(LABELS[m], fontsize=8.5)
    fig.suptitle(title + " (one line per user; black bar = median)")
    return fig
