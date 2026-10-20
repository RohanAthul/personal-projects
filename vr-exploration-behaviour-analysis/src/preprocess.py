"""Cleaning and resampling of raw traces into a gravity-aligned, metric, fixed-rate series.

Decisions (verified on the data, see notebook section 1):
- Head orientation is identical in the left- and right-eye rows of a frame; head position is
  taken as the midpoint of the two eye positions (each eye resampled separately, because two
  files do not strictly alternate left/right).
- Rotating poses by the scene's setting quaternion puts the head's up axis on world +Y; dividing
  positions by the scene scale gives physical metres (the paper's runtime scale factor).
- Forward viewing direction is local -Z (OpenXR convention).
- Samples whose timestamp does not increase are dropped.
- Resampling to a fixed rate; grid points inside a raw gap longer than `max_gap_s` are NaN.
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

QUAT_COLS = ["QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"]
POS_COLS = ["PositionX", "PositionY", "PositionZ"]


def monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose timestamp is strictly greater than every earlier timestamp."""
    t = df["Timestamp"].to_numpy()
    prev_max = np.maximum.accumulate(np.concatenate([[-np.inf], t[:-1]]))
    return df[t > prev_max]


def gap_mask(t_raw: np.ndarray, t_grid: np.ndarray, max_gap_s: float) -> np.ndarray:
    """True where a grid time lies inside the raw time range and its bracketing gap is <= max_gap_s."""
    inside = (t_grid >= t_raw[0]) & (t_grid <= t_raw[-1])
    idx = np.clip(np.searchsorted(t_raw, t_grid, side="right"), 1, len(t_raw) - 1)
    gap = t_raw[idx] - t_raw[idx - 1]
    return inside & (gap <= max_gap_s)


def resample_trace(raw: pd.DataFrame, scene_quat: np.ndarray, scale: float,
                   rate_hz: float = 30.0, duration_s: float = 60.0, max_gap_s: float = 0.2) -> pd.DataFrame:
    """Resample one raw trace to `rate_hz` over [0, duration_s).

    Returns columns t (s), qx, qy, qz, qw (gravity-aligned head orientation) and
    px, py, pz (head position in metres, +Y up). Invalid samples are NaN.
    """
    align = Rotation.from_quat(scene_quat)
    t_grid = np.arange(int(round(duration_s * rate_hz))) / rate_hz
    out = {"t": t_grid}

    # Orientation: one stream (left eye rows), slerp-interpolated.
    left = monotonic(raw[raw["ViewIndex"] == 0])
    t_l = left["Timestamp"].to_numpy() / 1000.0
    rot = align * Rotation.from_quat(left[QUAT_COLS].to_numpy())
    ok_q = gap_mask(t_l, t_grid, max_gap_s)
    q = np.full((len(t_grid), 4), np.nan)
    q[ok_q] = Slerp(t_l, rot)(t_grid[ok_q]).as_quat()
    out.update(qx=q[:, 0], qy=q[:, 1], qz=q[:, 2], qw=q[:, 3])

    # Position: midpoint of separately interpolated left and right eye positions.
    eyes, ok_p = [], ok_q.copy()
    for view in (0, 1):
        e = monotonic(raw[raw["ViewIndex"] == view])
        t_e = e["Timestamp"].to_numpy() / 1000.0
        p = align.apply(e[POS_COLS].to_numpy()) / scale
        eyes.append(np.column_stack([np.interp(t_grid, t_e, p[:, k]) for k in range(3)]))
        ok_p &= gap_mask(t_e, t_grid, max_gap_s)
    pos = (eyes[0] + eyes[1]) / 2
    pos[~ok_p] = np.nan
    out.update(px=pos[:, 0], py=pos[:, 1], pz=pos[:, 2])
    return pd.DataFrame(out)


def forward_vectors(series: pd.DataFrame) -> np.ndarray:
    """Unit viewing direction (local -Z rotated into the world) per sample; NaN rows stay NaN."""
    q = series[["qx", "qy", "qz", "qw"]].to_numpy()
    f = np.full((len(q), 3), np.nan)
    ok = ~np.isnan(q).any(axis=1)
    f[ok] = Rotation.from_quat(q[ok]).apply([0.0, 0.0, -1.0])
    return f
