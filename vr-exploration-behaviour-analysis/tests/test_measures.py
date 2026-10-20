"""Known-answer tests for entropy, velocity, direction binning and resampling."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.measures import (angular_speed, coverage, floor_bins, mask_above, shannon_entropy, sphere_bins,
                          translation_speed, yaw_pitch)
from src.preprocess import forward_vectors, resample_trace

RATE = 30.0


# --- entropy -----------------------------------------------------------------------------------
def test_entropy_uniform_is_log2_k():
    for k in (1, 2, 8, 648):
        assert shannon_entropy(np.ones(k)) == pytest.approx(np.log2(k))


def test_entropy_known_distribution_and_zeros():
    assert shannon_entropy([2, 1, 1, 0, 0]) == pytest.approx(1.5)
    assert shannon_entropy([0, 5, 0]) == 0.0
    assert np.isnan(shannon_entropy([0, 0]))


def test_uniform_sphere_directions_give_near_max_entropy():
    # Isotropic directions should fill equal-area bins evenly.
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    counts = np.bincount(sphere_bins(*yaw_pitch(v)), minlength=648)
    assert coverage(counts) == 1.0
    assert shannon_entropy(counts) == pytest.approx(np.log2(648), abs=0.01)
    assert counts.std() / counts.mean() < 0.05


def test_fixed_direction_has_zero_entropy():
    v = np.tile([0.0, 0.0, -1.0], (100, 1))
    counts = np.bincount(sphere_bins(*yaw_pitch(v)), minlength=648)
    assert shannon_entropy(counts) == 0.0
    assert coverage(counts) == pytest.approx(1 / 648)


# --- directions --------------------------------------------------------------------------------
def test_yaw_pitch_axes():
    yaw, pitch = yaw_pitch(np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
    np.testing.assert_allclose(yaw[:2], [0, 90], atol=1e-9)
    np.testing.assert_allclose(pitch, [0, 0, 90, 0], atol=1e-9)
    assert abs(yaw[3]) == pytest.approx(180)


# --- velocity ----------------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [[0, 1, 0], [1, 0, 0], [1, 1, 1]])
def test_constant_rotation_speed(axis):
    t = np.arange(0, 2, 1 / RATE)
    axis = np.array(axis, float) / np.linalg.norm(axis)
    q = Rotation.from_rotvec(np.radians(90.0) * t[:, None] * axis).as_quat()  # 90 deg/s
    np.testing.assert_allclose(angular_speed(q, 1 / RATE), 90.0, rtol=1e-6)


def test_angular_speed_ignores_quaternion_sign():
    t = np.arange(0, 1, 1 / RATE)
    q = Rotation.from_rotvec(np.radians(45.0) * t[:, None] * [0, 1, 0]).as_quat()
    q[::2] *= -1  # same rotations, flipped sign
    np.testing.assert_allclose(angular_speed(q, 1 / RATE), 45.0, rtol=1e-6)


def test_translation_speed_horizontal_and_3d():
    t = np.arange(0, 2, 1 / RATE)
    pos = np.column_stack([0.9 * t, 0.5 * t, 1.2 * t])  # horizontal 1.5 m/s, vertical 0.5 m/s
    np.testing.assert_allclose(translation_speed(pos, 1 / RATE), 1.5, rtol=1e-9)
    np.testing.assert_allclose(translation_speed(pos, 1 / RATE, horizontal=False),
                               np.sqrt(1.5**2 + 0.5**2), rtol=1e-9)


def test_floor_bins_single_point_and_spread():
    still = np.zeros((50, 3))
    assert shannon_entropy(np.bincount(floor_bins(still))) == 0.0
    # Four corners of a 1 m square, equally often -> 4 cells -> 2 bits.
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1.0]] * 25) + 0.1
    assert shannon_entropy(np.bincount(floor_bins(corners)[floor_bins(corners) >= 0])) == pytest.approx(2.0)


# --- resampling end to end ---------------------------------------------------------------------
def synthetic_raw(dt_ms=14, duration_s=10.0, yaw_deg_s=60.0, walk_m_s=0.5, gap=None):
    """Raw rows in the dataset's format: alternating eyes, constant yaw rate, straight walk in X."""
    t = np.arange(0, duration_s * 1000, dt_ms, dtype=float)
    if gap is not None:
        t = t[(t < gap[0]) | (t > gap[1])]
    q = Rotation.from_rotvec(np.radians(yaw_deg_s) * (t / 1000)[:, None] * [0, 1, 0]).as_quat()
    rows = []
    for view, offset in ((0, 0.0), (1, 3.0)):
        x = walk_m_s * (t + offset) / 1000
        ipd = -0.032 if view == 0 else 0.032
        rows.append(pd.DataFrame({
            "ViewIndex": view, "PositionX": x + ipd, "PositionY": 1.6, "PositionZ": 0.0,
            "QuaternionX": q[:, 0], "QuaternionY": q[:, 1], "QuaternionZ": q[:, 2], "QuaternionW": q[:, 3],
            "Timestamp": t + offset}))
    return pd.concat(rows).sort_values("Timestamp", kind="stable").reset_index(drop=True)


def test_resample_recovers_speeds():
    s = resample_trace(synthetic_raw(), np.array([0, 0, 0, 1.0]), 1.0, RATE, duration_s=9.0)
    q = s[["qx", "qy", "qz", "qw"]].to_numpy()
    ang = angular_speed(q, 1 / RATE)
    lin = translation_speed(s[["px", "py", "pz"]].to_numpy(), 1 / RATE)
    np.testing.assert_allclose(ang[~np.isnan(ang)], 60.0, rtol=1e-6)
    np.testing.assert_allclose(lin[~np.isnan(lin)], 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.nanmedian(s.px - 0.5 * s.t), 0.0, atol=1e-9)  # eye offsets cancel


def test_resample_applies_scale_and_alignment():
    # Scene scale 2: raw units move twice as far as metres. Scene tilt 30 deg about X is undone.
    tilt = Rotation.from_euler("x", 30, degrees=True)
    raw = synthetic_raw(yaw_deg_s=0.0)
    raw[["PositionX", "PositionY", "PositionZ"]] = tilt.inv().apply(raw[["PositionX", "PositionY", "PositionZ"]].to_numpy()) * 2.0
    raw[["QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"]] = (tilt.inv() * Rotation.from_quat(
        raw[["QuaternionX", "QuaternionY", "QuaternionZ", "QuaternionW"]].to_numpy())).as_quat()
    s = resample_trace(raw, tilt.as_quat(), 2.0, RATE, duration_s=9.0)
    lin = translation_speed(s[["px", "py", "pz"]].to_numpy(), 1 / RATE)
    np.testing.assert_allclose(lin[~np.isnan(lin)], 0.5, rtol=1e-6)
    np.testing.assert_allclose(forward_vectors(s)[1:], np.tile([0, 0, -1.0], (len(s) - 1, 1)), atol=1e-9)


def test_resample_masks_long_gaps():
    s = resample_trace(synthetic_raw(gap=(4000, 5000)), np.array([0, 0, 0, 1.0]), 1.0, RATE, duration_s=9.0)
    in_gap = (s.t > 4.02) & (s.t < 4.98)
    assert s.loc[in_gap, "qx"].isna().all() and s.loc[in_gap, "px"].isna().all()
    assert s.loc[(s.t > 1) & (s.t < 3.9), "qx"].notna().all()


def test_lagged_speeds_match_constant_motion_and_mask_outliers():
    t = np.arange(0, 2, 1 / RATE)
    q = Rotation.from_rotvec(np.radians(120.0) * t[:, None] * [0, 1, 0]).as_quat()
    np.testing.assert_allclose(angular_speed(q, 1 / RATE, lag=6), 120.0, rtol=1e-6)
    pos = np.column_stack([0.3 * t, 0 * t, 0.4 * t])
    np.testing.assert_allclose(translation_speed(pos, 1 / RATE, lag=6), 0.5, rtol=1e-9)
    assert len(angular_speed(q, 1 / RATE, lag=6)) == len(t) - 6
    np.testing.assert_array_equal(np.isnan(mask_above(np.array([1.0, 700.0, 5.0]), 600)), [False, True, False])


def test_lag_suppresses_rounding_noise():
    # A still head with components rounded to 3 decimals (as in the dataset) shows spurious
    # frame-to-frame motion; the 0.2 s lag reduces it.
    rng = np.random.default_rng(1)
    q = np.round(np.tile(Rotation.from_rotvec([0.1, 0.7, 0.05]).as_quat(), (600, 1))
                 + rng.uniform(-5e-4, 5e-4, (600, 4)), 3)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    assert np.median(angular_speed(q, 1 / RATE, lag=6)) < np.median(angular_speed(q, 1 / RATE)) / 4
