"""Exploration measures: angular / translation speed, equal-area direction bins, entropy, coverage."""
import numpy as np
import pandas as pd


def yaw_pitch(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw in [-180, 180) and pitch in [-90, 90] degrees for unit vectors with +Y up and -Z at yaw 0."""
    yaw = np.degrees(np.arctan2(forward[:, 0], -forward[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(forward[:, 1], -1.0, 1.0)))
    return yaw, pitch


def angular_speed(q: np.ndarray, dt: float, lag: int = 1) -> np.ndarray:
    """Head rotation speed (deg/s) between unit quaternions (x, y, z, w) `lag` samples apart;
    length n-lag.

    Uses the geodesic angle 2*arccos(|<q_i, q_{i+lag}>|), so it counts yaw, pitch and roll and is
    insensitive to the quaternion sign ambiguity. A lag > 1 suppresses rounding noise in the raw
    data at the cost of smoothing direction reversals faster than lag*dt.
    """
    dot = np.abs(np.sum(q[lag:] * q[:-lag], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))) / (lag * dt)


def translation_speed(pos: np.ndarray, dt: float, horizontal: bool = True, lag: int = 1) -> np.ndarray:
    """Head translation speed (m/s) between positions `lag` samples apart; length n-lag.
    Horizontal uses X and Z only."""
    p = pos[:, [0, 2]] if horizontal else pos
    return np.linalg.norm(p[lag:] - p[:-lag], axis=1) / (lag * dt)


def mask_above(x: np.ndarray, limit: float) -> np.ndarray:
    """Copy of x with values above `limit` (physiologically implausible) set to NaN."""
    return np.where(x > limit, np.nan, x)


def sphere_bins(yaw: np.ndarray, pitch: np.ndarray, n_yaw: int = 36, n_pitch: int = 18) -> np.ndarray:
    """Equal-area bin index per direction: uniform in yaw and in sin(pitch). NaN input -> -1."""
    ok = ~(np.isnan(yaw) | np.isnan(pitch))
    iy = np.floor((yaw[ok] + 180.0) / 360.0 * n_yaw).astype(int) % n_yaw
    ip = np.minimum(np.floor((np.sin(np.radians(pitch[ok])) + 1.0) / 2.0 * n_pitch).astype(int), n_pitch - 1)
    out = np.full(len(yaw), -1)
    out[ok] = ip * n_yaw + iy
    return out


FLOOR_CELL_M = 0.25
FLOOR_CELLS = 32  # per side: an 8 m x 8 m square, larger than the 3 m x 3 m play area


def floor_bins(pos: np.ndarray, cell_m: float = FLOOR_CELL_M, n_cells: int = FLOOR_CELLS) -> np.ndarray:
    """Floor-grid bin per sample (X/Z) on an n_cells x n_cells square centred on the trace's
    median position. Samples outside the square or NaN -> -1."""
    xz = pos[:, [0, 2]]
    ok = ~np.isnan(xz).any(axis=1)
    ij = np.zeros_like(xz, dtype=int)
    ij[ok] = np.floor((xz[ok] - np.median(xz[ok], axis=0)) / cell_m).astype(int) + n_cells // 2
    inside = ok & (ij >= 0).all(axis=1) & (ij < n_cells).all(axis=1)
    out = np.full(len(xz), -1)
    out[inside] = ij[inside, 0] * n_cells + ij[inside, 1]
    return out


def bin_counts(bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Sample count per bin, ignoring invalid (-1) entries."""
    return np.bincount(bins[bins >= 0], minlength=n_bins)


def shannon_entropy(counts: np.ndarray) -> float:
    """Plug-in Shannon entropy in bits of a count (or probability) vector. Empty -> NaN."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if total == 0:
        return np.nan
    p = c[c > 0] / total
    return float(-(p * np.log2(p)).sum())


def coverage(counts: np.ndarray) -> float:
    """Share of bins with at least one sample."""
    return float(np.count_nonzero(counts) / len(counts))


SPEED_LAG_S = 0.2
MAX_ANG_DEG_S = 600.0
MAX_TRANS_M_S = 2.0


def speeds(series: pd.DataFrame, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Angular (deg/s) and horizontal translation (m/s) speed over a SPEED_LAG_S baseline,
    with implausible values masked."""
    lag = max(1, int(round(SPEED_LAG_S * rate_hz)))
    q = series[["qx", "qy", "qz", "qw"]].to_numpy()
    pos = series[["px", "py", "pz"]].to_numpy()
    ang = mask_above(angular_speed(q, 1.0 / rate_hz, lag), MAX_ANG_DEG_S)
    trans = mask_above(translation_speed(pos, 1.0 / rate_hz, lag=lag), MAX_TRANS_M_S)
    return ang, trans


def trace_measures(series: pd.DataFrame, forward: np.ndarray, rate_hz: float,
                   n_yaw: int = 36, n_pitch: int = 18) -> dict:
    """Summary measures for one resampled trace (or a window of one)."""
    ang, trans = speeds(series, rate_hz)
    pos = series[["px", "py", "pz"]].to_numpy()
    yaw, pitch = yaw_pitch(forward)
    dir_counts = bin_counts(sphere_bins(yaw, pitch, n_yaw, n_pitch), n_yaw * n_pitch)
    pos_counts = bin_counts(floor_bins(pos), FLOOR_CELLS**2)
    return {
        "ang_speed_deg_s": float(np.nanmedian(ang)),
        "trans_speed_m_s": float(np.nanmedian(trans)),
        "dir_entropy_bits": shannon_entropy(dir_counts),
        "dir_coverage": coverage(dir_counts),
        "pos_entropy_bits": shannon_entropy(pos_counts),
        "n_valid": int(np.count_nonzero(~np.isnan(forward).any(axis=1))),
    }
