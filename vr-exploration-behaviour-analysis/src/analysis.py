"""Statistical analyses on the per-trace measure tables, plus checks behind preprocessing choices."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.measures import angular_speed, translation_speed
from src.preprocess import POS_COLS, QUAT_COLS, monotonic
from src.stats import friedman, holm, paired_test

MEASURES = ["ang_speed_deg_s", "trans_speed_m_s", "dir_entropy_bits", "dir_coverage", "pos_entropy_bits"]
BASELINE, CONDITION = "3DGS", "ZipNeRF"


def group_wide(measures: pd.DataFrame, m: str) -> pd.DataFrame:
    """User x group table of measure m, each cell the user's mean over that group's scenes."""
    return measures.groupby(["user", "group"])[m].mean().unstack("group").dropna()


def compare_groups(measures: pd.DataFrame) -> pd.DataFrame:
    """Primary analysis: ZipNeRF vs 3DGS baseline, paired within user, Holm across measures."""
    rows = [{"measure": m, **paired_test(w[BASELINE].to_numpy(), w[CONDITION].to_numpy())}
            for m in MEASURES for w in [group_wide(measures, m)]]
    return holm(pd.DataFrame(rows))


def compare_scenes(measures: pd.DataFrame) -> pd.DataFrame:
    """Secondary: Friedman across all 12 scenes (users with complete data), Holm across measures."""
    rows = []
    for m in MEASURES:
        wide = measures.pivot(index="user", columns="scene", values=m).dropna()
        rows.append({"measure": m, **friedman(wide)})
    return holm(pd.DataFrame(rows))


def phase_wide(windows: pd.DataFrame, m: str) -> pd.DataFrame:
    """User x phase (early/late) table, each cell the user's mean over scenes."""
    w = windows.dropna(subset=["phase"])
    return w.groupby(["user", "phase"])[m].mean().unstack("phase").dropna()


def compare_phases(windows: pd.DataFrame) -> pd.DataFrame:
    """Late (40-60 s) vs early (0-20 s), paired within user, Holm across measures."""
    rows = [{"measure": m, **paired_test(w["early"].to_numpy(), w["late"].to_numpy())}
            for m in MEASURES for w in [phase_wide(windows, m)]]
    return holm(pd.DataFrame(rows))


# --- checks behind preprocessing decisions -------------------------------------------------------
def frame_checks(raw: pd.DataFrame, scene_quat: np.ndarray, scale: float) -> dict:
    """Evidence for the coordinate conventions, on one raw trace (left-eye rows).

    head_up_dot_*: mean dot product of the head's local +Y with world +Y, without/with the scene
    rotation (1 = upright). walk_cos_minus_z: mean cosine between the horizontal head direction
    (local -Z) and the horizontal walking direction while walking (> 0.3 m/s).
    horizontal_range_m: X/Z extent after dividing by scale (should fit the 3 m x 3 m area).
    """
    left = monotonic(raw[raw["ViewIndex"] == 0])
    align = Rotation.from_quat(scene_quat)
    head = Rotation.from_quat(left[QUAT_COLS].to_numpy())
    pos = align.apply(left[POS_COLS].to_numpy()) / scale
    t = left["Timestamp"].to_numpy() / 1000.0
    v = np.diff(pos[:, [0, 2]], axis=0) / np.diff(t)[:, None]
    walking = np.linalg.norm(v, axis=1) > 0.3
    fwd = (align * head).apply([0, 0, -1.0])[1:, [0, 2]]
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    vdir = v[walking] / np.linalg.norm(v[walking], axis=1, keepdims=True)
    return {
        "head_up_dot_raw": float(head.apply([0, 1.0, 0])[:, 1].mean()),
        "head_up_dot_aligned": float((align * head).apply([0, 1.0, 0])[:, 1].mean()),
        "walk_cos_minus_z": float(np.mean(np.sum(fwd[walking] * vdir, axis=1))),
        "horizontal_range_m": float(np.ptp(pos[:, [0, 2]], axis=0).max()),
        "raw_horizontal_range_units": float(np.ptp(left[["PositionX", "PositionZ"]].to_numpy(), axis=0).max()),
    }


def rounding_step_speeds(rate_hz: float, lag: int, decimals: int = 3, min_scale: float = 0.36) -> dict:
    """Apparent speed produced by a single rounding step of the raw data, over one frame vs over
    the `lag`-sample baseline. A quaternion component step of 10**-decimals is up to ~2x that in
    rotation angle (radians); a position step is 10**-decimals scene units, i.e. /min_scale metres
    in the smallest-scale scene."""
    step = 10.0 ** -decimals
    ang_step_deg = np.degrees(2 * step)
    pos_step_m = step / min_scale
    dt = 1.0 / rate_hz
    return {"ang_frame_deg_s": ang_step_deg / dt, "ang_lag_deg_s": ang_step_deg / (lag * dt),
            "trans_frame_m_s": pos_step_m / dt, "trans_lag_m_s": pos_step_m / (lag * dt)}
