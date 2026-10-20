# VR Exploration Behaviour Analysis

Temporal and entropy-based analysis of 6-DoF head movement during free exploration of real-world VR scenes, using a public dataset (EyeNavGS). All numbers below are produced by [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) and stored in [`results/summary.json`](results/summary.json); figures are in [`figures/`](figures/).

## Motivation

How people explore an immersive environment (where they look, how much of the space they cover, how fast they turn and walk, and how this changes over time) is a basic behavioural signal for VR research, e.g. for studying how audio-visual content guides attention. This project builds a small, reproducible pipeline for such measures on 6-DoF head traces and applies it to one question: **does exploration differ between two groups of scenes, compared against a baseline group, and does it change from early to late in a session?**

## Data

**EyeNavGS, Rutgers subset** (Ding et al., 2025, [arXiv:2506.02380](https://arxiv.org/abs/2506.02380); [repository](https://github.com/symmru/EyeNavGS_Rutgers_Dataset), Apache-2.0, pinned to commit `7f264fd` by `data/download.sh`).

- 22 users x 12 scenes reconstructed with 3D Gaussian Splatting; every user explored every scene (within-subject).
- Meta Quest Pro, free standing navigation (walking and turning) in a 3 m x 3 m area.
- Per-frame CSV rows per eye: head position, head orientation (quaternion), eye gaze, field of view, timestamp (ms). Eye gaze was not used.
- **Conditions.** The paper states that 4 scenes (alameda, berlin, london, nyc) come from ZipNeRF and are "mainly indoor"; the other 8 come from the original 3DGS paper. The 3DGS group is used as the **baseline**, the ZipNeRF group as the comparison condition. The paper gives no per-scene indoor/outdoor labels, so none are assumed.

## Methods

**Preprocessing** (`src/preprocess.py`, `src/pipeline.py`). Coordinate conventions are not documented per file, so they were checked on all 264 traces (notebook §1):

| Decision | Evidence from the data |
|---|---|
| Rotate pose and position by the scene's setting quaternion to get a gravity-aligned frame (+Y up) | mean dot product of head-up with world-up: 0.89 raw, 0.95 aligned (lowest scene 0.91 after alignment) |
| Divide positions by the scene scale to get metres | raw horizontal range 0.88 to 5.5 units across scenes, 2.19 to 2.87 m after scaling (fits the 3 m area) |
| Viewing direction = local -Z | mean cosine with walking direction +0.43 (-Z), negative for +Z |
| Head position = midpoint of left/right eye positions | eye rows share the orientation; two files do not strictly alternate eyes, so each eye is resampled separately |

Cleaning steps:
1. Analyse the first 60 s of each trace (the protocol's one minute per scene), so all measures cover equal time. 7 of 264 traces are shorter and were excluded (257 kept).
2. Drop rows whose timestamp does not increase (1 row).
3. Resample to 30 Hz (median raw per-eye rate is 35.7 or 71.4 Hz, depending on the scene). Orientation uses slerp, position uses linear interpolation.
4. Leave grid points inside raw gaps longer than 0.2 s as missing (14 gaps in 13 traces; 526 of 462,600 samples).
5. Compute speeds over a 0.2 s baseline. The raw data are rounded to 3 decimals: one rounding step reads as 3.4 deg/s and 0.083 m/s frame-to-frame (worst-case scene scale 0.36), but only 0.57 deg/s and 0.014 m/s over 0.2 s.
6. Mask speeds above 600 deg/s or 2 m/s (14 angular and 60 translation samples of 460,448). Four traces contain single-step position jumps of 0.74 to 3.13 m (tracking glitches). They are kept, and the primary test is repeated without them.

**Measures** (`src/measures.py`), per user x scene:

| Measure | Definition |
|---|---|
| Angular speed | median geodesic head-rotation speed (deg/s) |
| Translation speed | median horizontal head speed (m/s) |
| Viewing-direction entropy | Shannon entropy (bits) of time per equal-area direction bin (36 yaw x 18 sin-pitch = 648 bins, max 9.34 bits) |
| Coverage | share of the 648 direction bins visited at all |
| Floor-position entropy | Shannon entropy (bits) of time per 0.25 m x 0.25 m floor cell |
| Dwell time per region | share of time per direction bin (Fig. 1) |

The same measures are computed in sliding 10 s windows (5 s step) and in an early (0-20 s) and late (40-60 s) phase.

**Statistics** (`src/stats.py`, `src/analysis.py`):
- **Pairing.** Users are the unit of analysis. For each user, measures are averaged over the scenes of each group (or phase), so all tests are paired.
- **Test choice.** Shapiro-Wilk on the paired differences decides the test: a paired t-test with Cohen's d_z if p ≥ .05, otherwise a Wilcoxon signed-rank test with the rank-biserial r. All differences passed Shapiro-Wilk, so all reported tests are paired t-tests.
- **Uncertainty.** 95% CIs come from a percentile bootstrap over users (10,000 resamples, fixed seed).
- **Multiple comparisons.** Holm correction is applied across the 5 measures within each family of tests.
- **Secondary test.** A Friedman test across all 12 scenes (complete cases, n = 17), with Kendall's W.

Known-answer tests on synthetic data cover the entropy, velocity, binning, resampling and effect-size functions (`tests/`, 20 tests).

## Results

**ZipNeRF vs 3DGS baseline** (n = 22 users; difference = ZipNeRF − 3DGS; [table](results/primary_zipnerf_vs_3dgs.csv), Fig. 3):

| Measure | Mean difference [95% CI] | d_z [95% CI] | p (Holm) |
|---|---|---|---|
| Translation speed (m/s) | +0.031 [0.017, 0.046] | 0.89 [0.47, 1.58] | 0.002 |
| Floor-position entropy (bits) | +0.25 [0.14, 0.37] | 0.88 [0.52, 1.39] | 0.002 |
| Angular speed (deg/s) | +0.15 [−1.26, 1.57] | 0.04 [−0.39, 0.52] | 1.0 |
| Viewing-direction entropy (bits) | −0.04 [−0.16, 0.08] | −0.13 [−0.61, 0.29] | 1.0 |
| Coverage (share of bins) | −0.010 [−0.021, 0.001] | −0.37 [−0.90, 0.04] | 0.29 |

- **What changed.** In the ZipNeRF scenes users walked faster and spread their time over more floor cells.
- **What did not.** How fast they turned their heads and how widely they distributed their gaze showed no detectable difference.
- **Glitch traces.** Both effects remain after removing the four glitch traces (Holm p = 0.002 for each).
- **Single-scene dependence.** Leaving out one ZipNeRF scene at a time, the floor-entropy effect stays significant in all four cases. The translation-speed effect does not survive leaving out nyc (Holm p = 0.070), so it depends partly on that one scene.
- **Scene scale.** Per-scene median translation speed is not clearly related to scene scale (Spearman ρ = −0.19, p = 0.55, 12 scenes). The smallest-scale scene, where rounding noise is largest, has the lowest speed.

**Scene effects** (Friedman, n = 17, 12 scenes): scenes differed in translation speed (χ² = 37.0, W = 0.20, Holm p < 0.001) and floor-position entropy (χ² = 41.0, W = 0.22, Holm p < 0.001). They did not differ in angular speed (W = 0.04), direction entropy (W = 0.07) or coverage (W = 0.11, Holm p = 0.086).

**Early (0-20 s) vs late (40-60 s)** (n = 22; difference = late − early; Fig. 2 and Fig. 4):
- **Angular speed** fell by 2.9 deg/s [1.1, 4.7] (d_z = −0.65 [−1.15, −0.27], Holm p = 0.032).
- **Direction entropy** fell by 0.17 bits [0.05, 0.29] (d_z = −0.57) and **coverage** by 0.010 [0.003, 0.016] (d_z = −0.58). Their bootstrap CIs exclude zero, but neither survives Holm correction (p = 0.052 for each).
- **Translation speed and floor-position entropy** showed no early/late change.

**Figures**
- [Fig. 1](figures/fig1_dwell_maps.png): dwell maps per scene, pooled over users.
- [Fig. 2](figures/fig2_time_course.png): entropy and speed in 10 s windows.
- [Fig. 3](figures/fig3_group_comparison.png): per-user group comparison.
- [Fig. 4](figures/fig4_early_vs_late.png): per-user early vs late.

**What the data can and cannot show.**
- **What it can show.** In this sample, the ZipNeRF scenes went with more walking and more spread-out floor positions, with no detectable change in head-turning or gaze spread. Across scenes, head rotation slowed from the first to the last 20 s.
- **Group differences are not causal.** The two groups differ in many ways at once: source dataset, scene layout (multi-room captures), rendering frame rate and runtime scale. The group difference is therefore not evidence for any single cause, including indoor vs outdoor.
- **Early vs late is not a clean time effect.** The early/late change could reflect habituation, the fixed starting viewpoint, or task disengagement; these cannot be separated here.

## Limitations

- **Secondary data.** I recruited no participants and had no control over design, instructions or scene order (the order is not reported in the paper).
- **No audio and no manipulated condition.** The comparison is between pre-existing scene groups, so it is observational, and there are only 4 scenes in the comparison group.
- **Small sample from one site.** 22 users from the Rutgers site; the NTHU subset was not used. The Friedman test used only the 17 users with all 12 scenes.
- **Head direction, not gaze.** Eye-tracking data exist but were not analysed.
- **Inferred conventions.** The coordinate conventions and the metre conversion were inferred from the data and the paper, not from author documentation. They pass consistency checks but are not guaranteed.
- **Measurement resolution.** Raw values are rounded to 3 decimals. The 0.2 s speed baseline reduces this noise but also smooths very fast head reversals.
- **Binning choices.** Entropy uses the plug-in estimator, which is biased low, and depends on the bin sizes (648 direction bins, 0.25 m floor cells). Comparisons use equal-length windows, but no other bin sizes were tested.
- **Yaw is scene-specific.** Yaw is measured in each scene's own frame, so dwell maps can be compared across users within a scene but not in absolute direction across scenes.
- **Analysis simplifications.** Per-user averaging over scenes ignores scene-level variance; a mixed-effects model would use the data more fully. Normality was checked with Shapiro-Wilk at n = 22, which has limited power.
- **Unreviewed source.** The dataset is described in a 2025 preprint.

## Future work

- **Audio-visual conditions.** Apply the same pipeline to 6-DoF data where audio is manipulated, for example spatial vs non-spatial audio, or sound sources placed inside vs outside the field of view.
- **A designed user study.** Run my own within-subject study with counterbalanced condition order, a pre-registered primary measure, and a power analysis informed by the effect sizes above.
- **Better models.** Use linear mixed-effects models with random effects for users and scenes.
- **Robustness and extensions.** Add bias-corrected entropy (e.g. Miller-Madow) and a bin-size sensitivity analysis, and include eye gaze.
- **Replication.** Repeat the analysis on the NTHU subset of EyeNavGS.

## How to run

Python 3.11+ (developed with 3.12.3).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./data/download.sh                      # clones the dataset (~330 MB with git history) into data/raw/
pytest                                   # 20 known-answer tests
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The notebook writes all tables to `results/` and figures to `figures/` (about 20 s end to end). Re-running gives identical results (fixed bootstrap seeds).

```
data/download.sh     pinned dataset download (raw data is git-ignored)
src/load.py          file listing, scene settings, scene groups
src/preprocess.py    alignment, metric scaling, gap-aware resampling
src/measures.py      speeds, equal-area binning, entropy, coverage
src/stats.py         paired tests, effect sizes, bootstrap CIs, Holm, Friedman
src/analysis.py      the comparisons and the convention checks
src/pipeline.py      raw traces -> measure tables
src/plots.py         figures
notebooks/           end-to-end analysis
tests/               synthetic known-answer tests
```

**Citation for the data:** Z. Ding, C.-T. Lee, M. Zhu, T. Guan, Y.-C. Sun, C.-H. Hsu, Y. Liu. *EyeNavGS: A 6-DoF Navigation Dataset and Record-n-Replay Software for Real-World 3DGS Scenes in VR.* arXiv:2506.02380, 2025.
