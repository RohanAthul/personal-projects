"""End-to-end processing: raw traces -> per-trace measures, time-window measures, dwell maps."""
import numpy as np
import pandas as pd

from src.load import list_traces, load_scene_settings, load_trace, scene_group
from src.measures import (MAX_ANG_DEG_S, MAX_TRANS_M_S, SPEED_LAG_S, angular_speed, bin_counts,
                          sphere_bins, trace_measures, translation_speed, yaw_pitch)
from src.preprocess import forward_vectors, resample_trace

RATE_HZ = 30.0
DURATION_S = 60.0
MAX_GAP_S = 0.2
WINDOW_S, STEP_S = 10.0, 5.0
EARLY, LATE = (0.0, 20.0), (40.0, 60.0)
N_YAW, N_PITCH = 36, 18
JUMP_M = 0.5  # single-step position jump flagged for the sensitivity analysis


def window(series: pd.DataFrame, fwd: np.ndarray, t0: float, t1: float):
    m = ((series["t"] >= t0) & (series["t"] < t1)).to_numpy()
    return series[m], fwd[m]


def process_trace(raw: pd.DataFrame, quat: np.ndarray, scale: float):
    """Resample one trace and return (series, forward, overall, windows, dwell_counts, qc)."""
    raw_t = raw["Timestamp"].to_numpy()
    s = resample_trace(raw, quat, scale, RATE_HZ, DURATION_S, MAX_GAP_S)
    f = forward_vectors(s)
    overall = trace_measures(s, f, RATE_HZ, N_YAW, N_PITCH)

    rows = []
    for t0 in np.arange(0, DURATION_S - WINDOW_S + 1e-9, STEP_S):
        rows.append({"t_mid_s": t0 + WINDOW_S / 2, **trace_measures(*window(s, f, t0, t0 + WINDOW_S), RATE_HZ)})
    for name, (t0, t1) in (("early", EARLY), ("late", LATE)):
        rows.append({"phase": name, **trace_measures(*window(s, f, t0, t1), RATE_HZ)})

    lag = int(round(SPEED_LAG_S * RATE_HZ))
    pos = s[["px", "py", "pz"]].to_numpy()
    ang_raw = angular_speed(s[["qx", "qy", "qz", "qw"]].to_numpy(), 1.0 / RATE_HZ, lag)
    trans_raw = translation_speed(pos, 1.0 / RATE_HZ, lag=lag)
    step = translation_speed(pos, 1.0 / RATE_HZ) / RATE_HZ
    raw_dt = np.diff(raw_t[(raw["ViewIndex"].to_numpy() == 0) & (raw_t < DURATION_S * 1000)])
    qc = {
        "raw_duration_s": raw_t.max() / 1000.0,
        "raw_dt_median_ms": float(np.median(raw_dt)),
        "non_monotonic_rows": int((raw_dt <= 0).sum()),
        "raw_gaps_over_max": int((raw_dt > MAX_GAP_S * 1000).sum()),
        "resampled_nan": int(np.isnan(f).any(axis=1).sum()),
        "ang_masked": int(np.sum(ang_raw > MAX_ANG_DEG_S)),
        "trans_masked": int(np.sum(trans_raw > MAX_TRANS_M_S)),
        "speed_samples": int(np.sum(~np.isnan(ang_raw))),
        "max_step_m": float(np.nanmax(step)),
    }
    dwell = bin_counts(sphere_bins(*yaw_pitch(f), N_YAW, N_PITCH), N_YAW * N_PITCH)
    return s, f, overall, pd.DataFrame(rows), dwell, qc


def run_all():
    """Process every trace >= DURATION_S long. Returns (measures, windows, dwell, qc) tables."""
    settings, traces = load_scene_settings(), list_traces()
    measures, windows, dwell, qc = [], [], {}, []
    for tr in traces.itertuples():
        raw = load_trace(tr.path)
        key = {"user": tr.user, "scene": tr.scene, "group": scene_group(tr.scene)}
        if raw["Timestamp"].max() / 1000.0 < DURATION_S:
            qc.append({**key, "included": False, "raw_duration_s": raw["Timestamp"].max() / 1000.0})
            continue
        _, _, overall, win, dw, q = process_trace(raw, settings.loc[tr.scene, "quat"], settings.loc[tr.scene, "scale"])
        measures.append({**key, **overall})
        windows.append(win.assign(**key))
        dwell[(tr.user, tr.scene)] = dw
        qc.append({**key, "included": True, **q, "position_jump": q["max_step_m"] > JUMP_M})
    return pd.DataFrame(measures), pd.concat(windows, ignore_index=True), dwell, pd.DataFrame(qc)
