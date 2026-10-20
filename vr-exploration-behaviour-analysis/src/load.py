"""Loading the EyeNavGS Rutgers traces and scene settings."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "EyeNavGS_Rutgers_Dataset"

# Scene source as stated in Ding et al. (2025), Sec. 3: four scenes come from ZipNeRF
# ("mainly indoor"), the remaining eight from the original 3DGS paper.
ZIPNERF_SCENES = {"alameda", "berlin", "london", "nyc"}


def scene_group(scene: str) -> str:
    """Return 'ZipNeRF' or '3DGS' (the baseline group) for a scene name."""
    return "ZipNeRF" if scene in ZIPNERF_SCENES else "3DGS"


def load_scene_settings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Scene table indexed by name, with `quat` (x, y, z, w) and `scale` columns."""
    s = pd.read_csv(data_dir / "scene_setting.csv", encoding="utf-8-sig")
    s["quat"] = s["Quaternion"].map(lambda v: np.array([float(x) for x in v.split(",")]))
    return s.rename(columns={"Scene_Name": "scene", "Scale": "scale"}).set_index("scene")[["quat", "scale"]]


def list_traces(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per trace file: user, scene, path."""
    rows = [
        {"user": p.stem.split("_")[0], "scene": p.parent.name, "path": p}
        for p in sorted((data_dir / "dataset").glob("*/*.csv"))
    ]
    return pd.DataFrame(rows)


def load_trace(path: Path) -> pd.DataFrame:
    """Raw per-eye rows of one trace (columns as documented in the dataset README)."""
    return pd.read_csv(path)
