#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 SSVEP same-frequency phase-transfer analysis.

The script is designed for MAT files produced by stage1_ssvep_phase_transfer.py.
It supports:
  * one or many sub* folders;
  * one or many MAT files in each subject folder;
  * fixed 8 EEG + 1 marker channel layout;
  * automatic parsing of trial_schedule / trial_log metadata;
  * configurable block count and trials per phase per block;
  * multiple analysis-window lengths;
  * phase-transfer, phase-error, PLV, classification and capacity analyses;
  * CSV tables, PNG figures and a Markdown summary report.

Important convention
--------------------
The trial phase estimator fits
    x(t) = a*sin(2*pi*f*t) + b*cos(2*pi*f*t) + c + d*t
and returns angle(a + 1j*b). For x(t)=A*sin(2*pi*f*t+phi), this angle is phi.

Recommended first run
---------------------
python stage1_phase_transfer_analysis.py \
    --root /path/to/data \
    --output /path/to/stage1_analysis

Channel indices are Python-style, zero based.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from scipy.stats import kurtosis, levene, norm, probplot, shapiro, skew, vonmises


# =============================================================================
# 1. User-editable defaults
# =============================================================================


DEFAULT_CHANNEL_NAMES = ("FCz", "TP9", "Pz", "POz", "O1", "Oz", "O2", "TP10", "channel")
DEFAULT_EEG_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_MARKER_CHANNEL = 8
DEFAULT_REFERENCE_CHANNEL = 5  # Oz
LINKED_MASTOID_CHANNELS = (1, 7)  # TP9, TP10


@dataclass
class AnalysisConfig:
    # File discovery
    root_dir: Path = Path(".")
    output_dir: Path = Path("stage1_analysis")
    subject_glob: str = "sub*"
    file_glob: str = "Stage1_SSVEP_PhaseTransfer_*.mat"
    recursive: bool = True

    # Optional experiment-parameter overrides. None means read from each MAT file.
    fs_override: Optional[float] = None
    stim_freq_override: Optional[float] = None
    phase_count_override: Optional[int] = None
    phase_values_deg_override: Optional[tuple[float, ...]] = None
    trigger_base_override: Optional[int] = None

    # Trial/block subset. None means use all valid recorded trials.
    # When integers are supplied, only the first N blocks and first N trials in
    # each (file, block, phase) cell are retained.
    block_count: Optional[int] = None
    trials_per_phase_per_block: Optional[int] = None

    # Data channels. Indices are zero based for the fixed layout:
    # 0 FCz, 1 TP9, 2 Pz, 3 POz, 4 O1, 5 Oz, 6 O2, 7 TP10, 8 marker.
    marker_channel: int = DEFAULT_MARKER_CHANNEL
    eeg_channels: tuple[int, ...] = DEFAULT_EEG_CHANNELS
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES
    reference_channel: int = DEFAULT_REFERENCE_CHANNEL
    spatial_mode: str = "phase_aligned"  # single | mean | phase_aligned
    rereference: str = "none"             # none | average
    linked_mastoids_reference: bool = False

    # Event alignment
    marker_integer_tolerance: float = 0.20
    event_match_tolerance_sec: float = 0.080
    min_inter_event_sec: float = 0.100
    allow_timing_fallback: bool = False
    manual_first_onset_sec: Optional[float] = None
    min_event_match_fraction: float = 0.80

    # Signal processing
    transient_sec: float = 0.30
    window_lengths_sec: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 45.0
    filter_order: int = 4
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0

    # Artifact flagging. Primary analysis keeps all trials unless reject_artifacts=True.
    artifact_mad_threshold: float = 8.0
    absolute_peak_to_peak_limit: Optional[float] = None
    reject_artifacts: bool = False

    # Statistics
    classification_window_sec: float = 1.0
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 2000
    random_seed: int = 20260714
    alpha: float = 0.05
    capacity_error_threshold: float = 0.10
    capacity_max_m: int = 30

    # Configurable descriptive decision thresholds. These are not universal laws.
    slope_tolerance: float = 0.15
    theta_range_tolerance_deg: float = 15.0
    weak_locking_sigma_deg: float = 40.0
    min_trials_per_phase_for_confirmatory: int = 30

    # Output
    save_trial_channel_metrics: bool = True
    plot_dpi: int = 180

    def validate(self) -> None:
        if self.spatial_mode not in {"single", "mean", "phase_aligned"}:
            raise ValueError("spatial_mode must be single, mean, or phase_aligned")
        if self.rereference not in {"none", "average"}:
            raise ValueError("rereference must be none or average")
        if self.reference_channel not in self.eeg_channels:
            raise ValueError("reference_channel must be included in eeg_channels")
        if self.marker_channel in self.eeg_channels:
            raise ValueError("marker_channel cannot be included in eeg_channels")
        if not self.window_lengths_sec:
            raise ValueError("window_lengths_sec cannot be empty")
        if any(w <= 0 for w in self.window_lengths_sec):
            raise ValueError("all analysis windows must be > 0")
        if self.transient_sec < 0:
            raise ValueError("transient_sec must be >= 0")
        if self.bootstrap_iterations < 0 or self.permutation_iterations < 0:
            raise ValueError("iteration counts must be >= 0")
        if not (0 < self.capacity_error_threshold < 1):
            raise ValueError("capacity_error_threshold must be between 0 and 1")
        if self.block_count is not None and self.block_count < 1:
            raise ValueError("block_count must be >= 1 or None")
        if self.trials_per_phase_per_block is not None and self.trials_per_phase_per_block < 1:
            raise ValueError("trials_per_phase_per_block must be >= 1 or None")


# =============================================================================
# 2. Circular-statistics and utility functions
# =============================================================================


def wrap_rad(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to (-pi, pi]."""
    y = (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi
    y = np.where(np.isclose(y, -np.pi), np.pi, y)
    return float(y) if np.ndim(x) == 0 else y


def circ_mean(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.angle(np.mean(np.exp(1j * a))))


def circ_r(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * a))))


def circ_std(x: Sequence[float]) -> float:
    r = circ_r(x)
    if not np.isfinite(r):
        return float("nan")
    r = float(np.clip(r, 1e-12, 1.0))
    return float(np.sqrt(-2.0 * np.log(r)))


def circ_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs(wrap_rad(np.asarray(a) - np.asarray(b)))


def circular_peak_to_peak(angles: Sequence[float], center: Optional[float] = None) -> float:
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if center is None:
        center = circ_mean(a)
    u = center + wrap_rad(a - center)
    return float(np.max(u) - np.min(u))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def wrapped_normal_pdf(x: np.ndarray, sigma: float, k_max: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.full_like(x, np.nan)
    out = np.zeros_like(x)
    for k in range(-k_max, k_max + 1):
        out += norm.pdf(x + 2.0 * np.pi * k, loc=0.0, scale=sigma)
    return out


def wrapped_normal_correct_probability(half_width: float, sigma: float, k_max: int = 12) -> float:
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0
    p = 0.0
    for k in range(-k_max, k_max + 1):
        lower = (-half_width + 2.0 * np.pi * k) / sigma
        upper = (half_width + 2.0 * np.pi * k) / sigma
        p += norm.cdf(upper) - norm.cdf(lower)
    return float(np.clip(p, 0.0, 1.0))


def predicted_error_gaussian(m: int, sigma: float) -> float:
    if m < 2 or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(np.clip(2.0 * norm.sf(np.pi / (m * sigma)), 0.0, 1.0))


def predicted_error_wrapped_normal(m: int, sigma: float) -> float:
    if m < 2 or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return 1.0 - wrapped_normal_correct_probability(np.pi / m, sigma)


def mmax_closed_form(sigma: float, epsilon: float) -> int:
    if not np.isfinite(sigma) or sigma <= 0:
        return 0
    z = norm.ppf(1.0 - epsilon / 2.0)
    return max(1, int(np.floor(np.pi / (z * sigma))))


def mmax_wrapped_normal(sigma: float, epsilon: float, max_m: int) -> int:
    valid = [m for m in range(2, max_m + 1) if predicted_error_wrapped_normal(m, sigma) <= epsilon]
    return max(valid) if valid else 1


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return default
        return float(arr.flat[0])
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return default
        return int(round(float(arr.flat[0])))
    except Exception:
        return default


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object).ravel()
    out: list[str] = []
    for item in arr:
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        out.append(str(item))
    return out


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def robust_mad(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    med = np.nanmedian(a)
    return float(1.4826 * np.nanmedian(np.abs(a - med)))


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


# =============================================================================
# 3. MAT-file discovery and metadata parsing
# =============================================================================


def discover_subject_files(cfg: AnalysisConfig) -> dict[str, list[Path]]:
    root = cfg.root_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    subjects: dict[str, list[Path]] = {}
    subject_dirs = sorted([p for p in root.glob(cfg.subject_glob) if p.is_dir()], key=lambda p: natural_key(p.name))

    if subject_dirs:
        for subdir in subject_dirs:
            iterator = subdir.rglob(cfg.file_glob) if cfg.recursive else subdir.glob(cfg.file_glob)
            files = sorted([p for p in iterator if p.is_file()], key=lambda p: natural_key(str(p)))
            if files:
                subjects[subdir.name] = files
    else:
        iterator = root.rglob(cfg.file_glob) if cfg.recursive else root.glob(cfg.file_glob)
        files = sorted([p for p in iterator if p.is_file()], key=lambda p: natural_key(str(p)))
        if files:
            subjects[root.name or "subject"] = files

    if not subjects:
        raise FileNotFoundError(
            f"No MAT files found under {root} using subject_glob={cfg.subject_glob!r}, "
            f"file_glob={cfg.file_glob!r}."
        )
    return subjects


def load_mat_file(path: Path) -> dict[str, Any]:
    try:
        return loadmat(path, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def orient_data_matrix(data: Any) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D after squeeze, got shape={x.shape}")

    # Most EEG recordings have many more samples than channels.
    if x.shape[0] <= 64 and x.shape[1] > x.shape[0] * 4:
        x = x.T
    elif x.shape[1] <= 64 and x.shape[0] > x.shape[1] * 4:
        pass
    elif x.shape[0] < x.shape[1]:
        x = x.T

    if x.shape[1] > 256:
        raise ValueError(f"Could not identify channel axis safely, oriented shape={x.shape}")
    return x


def parse_matrix_with_columns(mat: dict[str, Any], matrix_key: str, columns_key: str) -> pd.DataFrame:
    if matrix_key not in mat:
        return pd.DataFrame()
    values = np.asarray(mat[matrix_key])
    if values.size == 0:
        return pd.DataFrame()
    if values.ndim == 1:
        values = values[None, :]
    columns = _flatten_strings(mat.get(columns_key))
    if len(columns) != values.shape[1]:
        columns = [f"col_{i}" for i in range(values.shape[1])]
    df = pd.DataFrame(values, columns=columns)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass
    return df


def build_expected_trials(mat: dict[str, Any], cfg: AnalysisConfig, detected_events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    schedule = parse_matrix_with_columns(mat, "trial_schedule", "trial_schedule_columns")
    trial_log = parse_matrix_with_columns(mat, "trial_log", "trial_log_columns")

    if not schedule.empty:
        rename = {c: c.strip() for c in schedule.columns}
        schedule = schedule.rename(columns=rename)
        if "trial_idx" not in schedule.columns:
            schedule["trial_idx"] = np.arange(len(schedule))
        if not trial_log.empty and "trial_idx" in trial_log.columns:
            log_cols = [c for c in ["trial_idx", "onset_time_sec", "trigger", "onset_frame"] if c in trial_log.columns]
            trial_log = trial_log[log_cols].copy()
            # The log contains trials that were actually presented. Inner join protects aborted runs.
            schedule = schedule.merge(trial_log, on="trial_idx", how="inner", suffixes=("", "_log"))
            if "trigger_log" in schedule.columns:
                schedule["trigger"] = schedule["trigger_log"]
            if "onset_frame_log" in schedule.columns:
                schedule["onset_frame_actual"] = schedule["onset_frame_log"]
        else:
            schedule["onset_time_sec"] = np.nan

        required_defaults = {
            "block_idx": 0,
            "trial_in_block": np.arange(len(schedule)),
            "phase_idx": np.nan,
            "phase_deg": np.nan,
            "trigger": np.nan,
            "label": np.nan,
        }
        for col, default in required_defaults.items():
            if col not in schedule.columns:
                schedule[col] = default

        return schedule.sort_values("trial_idx").reset_index(drop=True)

    # Fallback for old MAT files without schedule metadata.
    if detected_events is None or detected_events.empty:
        raise ValueError("MAT file has no trial_schedule and no marker events from which to reconstruct trials")

    trigger_base = cfg.trigger_base_override
    if trigger_base is None:
        trigger_base = int(detected_events["event_label"].min())

    phase_count = cfg.phase_count_override
    if phase_count is None:
        phase_count = int(detected_events["event_label"].nunique())

    if cfg.phase_values_deg_override is not None:
        phases_deg = np.asarray(cfg.phase_values_deg_override, dtype=float)
    else:
        phases_deg = np.arange(phase_count, dtype=float) * 360.0 / phase_count

    rows = []
    trials_per_block = None
    if cfg.trials_per_phase_per_block is not None:
        trials_per_block = phase_count * cfg.trials_per_phase_per_block

    for i, event in detected_events.reset_index(drop=True).iterrows():
        label = int(event["event_label"])
        phase_idx = label - trigger_base
        if phase_idx < 0 or phase_idx >= len(phases_deg):
            continue
        block_idx = 0 if trials_per_block is None else i // trials_per_block
        rows.append({
            "trial_idx": i,
            "block_idx": block_idx,
            "trial_in_block": i if trials_per_block is None else i % trials_per_block,
            "label": phase_idx + 1,
            "phase_idx": phase_idx,
            "phase_deg": phases_deg[phase_idx],
            "phase_rad": np.deg2rad(phases_deg[phase_idx]),
            "trigger": label,
            "onset_time_sec": np.nan,
        })
    return pd.DataFrame(rows)


def infer_experiment_params(mat: dict[str, Any], schedule: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    fs = cfg.fs_override if cfg.fs_override is not None else _safe_float(mat.get("Fs"))
    stim_freq = cfg.stim_freq_override if cfg.stim_freq_override is not None else _safe_float(mat.get("stim_freq"))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Missing or invalid Fs; use --fs-override")
    if not np.isfinite(stim_freq) or stim_freq <= 0:
        raise ValueError("Missing or invalid stim_freq; use --stim-freq-override")

    phase_count = cfg.phase_count_override
    if phase_count is None:
        phase_count = _safe_int(mat.get("phase_count"), None)
    if phase_count is None and not schedule.empty:
        phase_count = int(schedule["phase_idx"].nunique())

    if cfg.phase_values_deg_override is not None:
        phases_deg = np.asarray(cfg.phase_values_deg_override, dtype=float)
    elif "phases_deg" in mat:
        phases_deg = np.asarray(mat["phases_deg"], dtype=float).ravel()
    elif not schedule.empty and "phase_deg" in schedule.columns:
        phases_deg = np.sort(schedule["phase_deg"].dropna().astype(float).unique())
    elif phase_count is not None:
        phases_deg = np.arange(phase_count, dtype=float) * 360.0 / phase_count
    else:
        raise ValueError("Cannot infer phase values; use --phase-values-deg or --phase-count-override")

    if phase_count is None:
        phase_count = len(phases_deg)
    if len(phases_deg) != phase_count:
        raise ValueError(f"phase_count={phase_count} but phases_deg has {len(phases_deg)} entries")

    trigger_base = cfg.trigger_base_override
    if trigger_base is None:
        trigger_base = _safe_int(mat.get("trigger_base"), None)
    if trigger_base is None and not schedule.empty and "trigger" in schedule.columns:
        trigger_base = int(np.nanmin(schedule["trigger"]))
    if trigger_base is None:
        trigger_base = 1

    stimulus_duration = _safe_float(mat.get("stimulus_duration_sec"), float("nan"))
    return {
        "fs": float(fs),
        "stim_freq": float(stim_freq),
        "phase_count": int(phase_count),
        "phases_deg": phases_deg.astype(float),
        "trigger_base": int(trigger_base),
        "stimulus_duration_sec": stimulus_duration,
    }


# =============================================================================
# 4. Marker detection and trial-to-sample alignment
# =============================================================================


def extract_marker_events(
    marker: np.ndarray,
    expected_labels: Sequence[int],
    fs: float,
    integer_tolerance: float,
    min_inter_event_sec: float,
) -> pd.DataFrame:
    x = np.asarray(marker, dtype=float).ravel()
    finite = np.isfinite(x)
    rounded = np.rint(np.where(finite, x, 0.0))
    rounded_int = rounded.astype(np.int64, copy=False)
    is_integer = finite & (np.abs(x - rounded) <= integer_tolerance)
    expected_labels_arr = np.asarray(sorted(set(int(v) for v in expected_labels)), dtype=int)
    is_expected = is_integer & np.isin(rounded_int, expected_labels_arr)

    # Rising into a label, or transition from a different value into that label.
    prev_expected = np.r_[False, is_expected[:-1]]
    prev_value = np.r_[np.nan, rounded[:-1]]
    onset_mask = is_expected & ((~prev_expected) | (rounded != prev_value))
    candidates = np.flatnonzero(onset_mask)

    min_gap = max(1, int(round(min_inter_event_sec * fs)))
    keep: list[int] = []
    for idx in candidates:
        if not keep or idx - keep[-1] >= min_gap:
            keep.append(int(idx))
        elif rounded[idx] != rounded[keep[-1]]:
            # Two different labels too close together usually indicates a noisy marker channel.
            # Keep the earliest event and let alignment/QC reveal the mismatch.
            continue

    return pd.DataFrame({
        "event_sample": keep,
        "event_time_sec": np.asarray(keep, dtype=float) / fs,
        "event_label": [int(rounded[i]) for i in keep],
    })


def normalize_channel_index(ch: int, n_channels: int, role: str) -> int:
    idx = ch if ch >= 0 else n_channels + ch
    if idx < 0 or idx >= n_channels:
        raise IndexError(f"{role}={ch} is invalid for data with {n_channels} columns")
    return idx


def resolve_marker_channel(
    data: np.ndarray,
    expected_labels: Sequence[int],
    fs: float,
    cfg: AnalysisConfig,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    marker_channel = normalize_channel_index(cfg.marker_channel, data.shape[1], "marker_channel")
    events = extract_marker_events(
        data[:, marker_channel],
        expected_labels,
        fs,
        cfg.marker_integer_tolerance,
        cfg.min_inter_event_sec,
    )
    qc = pd.DataFrame([{
        "channel": marker_channel,
        "channel_name": DEFAULT_CHANNEL_NAMES[marker_channel] if marker_channel < len(DEFAULT_CHANNEL_NAMES) else f"Ch{marker_channel}",
        "event_count": len(events),
        "selected": True,
    }])
    return marker_channel, events, qc


def lcs_label_mapping(expected: Sequence[int], observed: Sequence[int]) -> dict[int, int]:
    """Map expected indices to observed indices using longest common subsequence."""
    a = list(map(int, expected))
    b = list(map(int, observed))
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int16 if max(n, m) < 30000 else np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return mapping


def match_by_timing(
    expected: pd.DataFrame,
    events: pd.DataFrame,
    fs: float,
    tolerance_sec: float,
) -> tuple[dict[int, int], Optional[float], float]:
    if expected.empty or events.empty or "onset_time_sec" not in expected.columns:
        return {}, None, float("inf")
    valid_expected = expected[np.isfinite(pd.to_numeric(expected["onset_time_sec"], errors="coerce"))]
    if valid_expected.empty:
        return {}, None, float("inf")

    exp_times = expected["onset_time_sec"].to_numpy(dtype=float)
    exp_labels = expected["trigger"].to_numpy(dtype=int)
    evt_times = events["event_time_sec"].to_numpy(dtype=float)
    evt_labels = events["event_label"].to_numpy(dtype=int)

    candidates: list[float] = []
    for i in range(len(expected)):
        if not np.isfinite(exp_times[i]):
            continue
        matching = np.flatnonzero(evt_labels == exp_labels[i])
        candidates.extend((evt_times[matching] - exp_times[i]).tolist())
    if not candidates:
        return {}, None, float("inf")

    tol = tolerance_sec
    best_mapping: dict[int, int] = {}
    best_offset = None
    best_cost = float("inf")

    # Deduplicate nearly identical offsets to keep the search small.
    candidates = sorted(set(round(c, 4) for c in candidates))
    for offset in candidates:
        mapping: dict[int, int] = {}
        used: set[int] = set()
        last_evt = -1
        residuals: list[float] = []
        for i in range(len(expected)):
            if not np.isfinite(exp_times[i]):
                continue
            pred = exp_times[i] + offset
            candidates_evt = np.flatnonzero((evt_labels == exp_labels[i]) & (np.arange(len(events)) > last_evt))
            if candidates_evt.size == 0:
                continue
            distances = np.abs(evt_times[candidates_evt] - pred)
            order = np.argsort(distances)
            chosen = None
            for pos in order:
                j = int(candidates_evt[pos])
                if j not in used and distances[pos] <= tol:
                    chosen = j
                    residuals.append(float(distances[pos]))
                    break
            if chosen is not None:
                mapping[i] = chosen
                used.add(chosen)
                last_evt = chosen
        unmatched = len(expected) - len(mapping)
        rms = math.sqrt(np.mean(np.square(residuals))) if residuals else 999.0
        cost = unmatched * 10.0 + rms
        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping
            best_offset = float(offset)

    rms = float("inf")
    if best_mapping and best_offset is not None:
        residuals = [
            evt_times[j] - (exp_times[i] + best_offset)
            for i, j in best_mapping.items()
            if np.isfinite(exp_times[i])
        ]
        rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    return best_mapping, best_offset, rms


def align_trials_to_samples(
    expected: pd.DataFrame,
    events: pd.DataFrame,
    fs: float,
    cfg: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = expected.copy().reset_index(drop=True)
    events = events.copy().reset_index(drop=True)
    expected_labels = expected["trigger"].astype(int).tolist()
    observed_labels = events["event_label"].astype(int).tolist()

    mapping: dict[int, int] = {}
    method = "none"
    offset = None
    timing_rms = float("nan")

    if len(expected) == len(events) and expected_labels == observed_labels:
        mapping = {i: i for i in range(len(expected))}
        method = "exact_sequence"
    else:
        timing_mapping, offset, timing_rms = match_by_timing(
            expected, events, fs, cfg.event_match_tolerance_sec
        )
        lcs_mapping = lcs_label_mapping(expected_labels, observed_labels)
        if len(timing_mapping) >= len(lcs_mapping) and timing_mapping:
            mapping = timing_mapping
            method = "timing_and_label"
        elif lcs_mapping:
            mapping = lcs_mapping
            method = "label_sequence_lcs"

    expected["onset_sample"] = np.nan
    expected["alignment_source"] = "unmatched"
    expected["matched_event_label"] = np.nan
    for i, j in mapping.items():
        expected.loc[i, "onset_sample"] = int(events.loc[j, "event_sample"])
        expected.loc[i, "alignment_source"] = method
        expected.loc[i, "matched_event_label"] = int(events.loc[j, "event_label"])

    if cfg.allow_timing_fallback:
        if offset is None and cfg.manual_first_onset_sec is not None:
            first_log = pd.to_numeric(expected["onset_time_sec"], errors="coerce").dropna()
            if not first_log.empty:
                offset = cfg.manual_first_onset_sec - float(first_log.iloc[0])
        if offset is not None and "onset_time_sec" in expected.columns:
            missing = expected["onset_sample"].isna()
            times = pd.to_numeric(expected.loc[missing, "onset_time_sec"], errors="coerce")
            valid = times.notna()
            idxs = times.index[valid]
            expected.loc[idxs, "onset_sample"] = np.rint((times.loc[idxs] + offset) * fs).astype(int)
            expected.loc[idxs, "alignment_source"] = "timing_fallback"

    matched = int(expected["onset_sample"].notna().sum())
    match_fraction = matched / max(1, len(expected))
    qc = {
        "alignment_method": method,
        "expected_trial_count": int(len(expected)),
        "detected_event_count": int(len(events)),
        "matched_trial_count": matched,
        "match_fraction": match_fraction,
        "estimated_recording_offset_sec": offset,
        "timing_match_rms_sec": timing_rms,
    }
    return expected, qc


def apply_trial_subset(trials: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = trials.copy()
    if cfg.block_count is not None and "block_idx" in out.columns:
        blocks = sorted(out["block_idx"].dropna().astype(int).unique())[: cfg.block_count]
        out = out[out["block_idx"].astype(int).isin(blocks)]
    if cfg.trials_per_phase_per_block is not None:
        group_cols = [c for c in ["block_idx", "phase_idx"] if c in out.columns]
        if group_cols:
            out = (
                out.sort_values("trial_idx")
                .groupby(group_cols, group_keys=False, observed=True)
                .head(cfg.trials_per_phase_per_block)
            )
    return out.sort_values("trial_idx").reset_index(drop=True)


# =============================================================================
# 5. Signal preprocessing and trial phase estimation
# =============================================================================


def resolve_channel_indices(
    n_channels: int,
    marker_channel: int,
    cfg: AnalysisConfig,
) -> tuple[list[int], list[str], int]:
    eeg = [normalize_channel_index(ch, n_channels, "EEG channel") for ch in cfg.eeg_channels]
    if marker_channel in eeg:
        raise ValueError(f"EEG channel list includes marker channel {marker_channel}")
    if not eeg:
        raise ValueError("No EEG channels remain after excluding the marker channel")

    if len(cfg.channel_names) == n_channels:
        names = [cfg.channel_names[idx] for idx in eeg]
    elif len(cfg.channel_names) == len(eeg):
        names = list(cfg.channel_names)
    else:
        raise ValueError(
            "channel_names must have either one name per data column or one name per selected EEG channel"
        )

    reference_abs = normalize_channel_index(cfg.reference_channel, n_channels, "reference_channel")
    if reference_abs not in eeg:
        raise ValueError(
            f"reference_channel={cfg.reference_channel} must be included in selected EEG channels {eeg}"
        )
    reference_pos = eeg.index(reference_abs)
    return eeg, names, reference_pos


def extract_eeg_matrix(data: np.ndarray, eeg_indices: Sequence[int], cfg: AnalysisConfig) -> np.ndarray:
    eeg = np.asarray(data[:, eeg_indices], dtype=float).copy()
    if not cfg.linked_mastoids_reference:
        return eeg

    n_channels = data.shape[1]
    mastoid_indices = [
        normalize_channel_index(ch, n_channels, "linked mastoid channel")
        for ch in LINKED_MASTOID_CHANNELS
    ]
    mastoids = interpolate_nonfinite(np.asarray(data[:, mastoid_indices], dtype=float))
    linked_ref = np.mean(mastoids, axis=1, keepdims=True)
    return eeg - linked_ref


def interpolate_nonfinite(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    n_samples, n_channels = y.shape
    idx = np.arange(n_samples)
    for ch in range(n_channels):
        finite = np.isfinite(y[:, ch])
        if finite.all():
            continue
        if finite.sum() < 2:
            y[:, ch] = 0.0
        else:
            y[~finite, ch] = np.interp(idx[~finite], idx[finite], y[finite, ch])
    return y


def preprocess_continuous(eeg: np.ndarray, fs: float, cfg: AnalysisConfig) -> np.ndarray:
    x = interpolate_nonfinite(eeg)
    if cfg.rereference == "average" and x.shape[1] > 1:
        x = x - np.mean(x, axis=1, keepdims=True)

    nyq = fs / 2.0
    if cfg.notch_hz is not None and 0 < cfg.notch_hz < nyq:
        b, a = iirnotch(cfg.notch_hz / nyq, cfg.notch_q)
        x = filtfilt(b, a, x, axis=0)

    low = max(0.01, cfg.bandpass_low_hz)
    high = min(cfg.bandpass_high_hz, nyq * 0.95)
    if low >= high:
        raise ValueError(f"Invalid bandpass [{low}, {high}] for Fs={fs}")
    sos = butter(cfg.filter_order, [low, high], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=0)
    return x


def harmonic_regression_coefficients(epoch: np.ndarray, fs: float, frequency: float) -> np.ndarray:
    """Return one complex sine/cosine coefficient per channel."""
    y = np.asarray(epoch, dtype=float)
    n = y.shape[0]
    t = np.arange(n, dtype=float) / fs
    tc = t - np.mean(t)
    design = np.column_stack([
        np.sin(2.0 * np.pi * frequency * t),
        np.cos(2.0 * np.pi * frequency * t),
        np.ones(n),
        tc,
    ])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta[0, :] + 1j * beta[1, :]


def regression_snr_db(epoch: np.ndarray, fs: float, frequency: float) -> np.ndarray:
    target = harmonic_regression_coefficients(epoch, fs, frequency)
    duration = epoch.shape[0] / fs
    df = 1.0 / max(duration, 1e-9)
    side_freqs = []
    for k in (2, 3, 4, 5):
        for sign in (-1, 1):
            f = frequency + sign * k * df
            if f > 0.5 and f < fs / 2.0 - 0.5:
                side_freqs.append(f)
    if not side_freqs:
        return np.full(epoch.shape[1], np.nan)
    side_power = np.stack([
        np.abs(harmonic_regression_coefficients(epoch, fs, f)) ** 2 for f in side_freqs
    ])
    noise_power = np.nanmedian(side_power, axis=0)
    target_power = np.abs(target) ** 2
    return 10.0 * np.log10((target_power + 1e-20) / (noise_power + 1e-20))


def combine_channel_coefficients(
    z: np.ndarray,
    snr_db: np.ndarray,
    mode: str,
    reference_pos: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    z shape: trials x channels.
    Returns combined complex coefficient, combined SNR and channel-alignment QC.
    """
    if mode == "single":
        combined = z[:, reference_pos]
        combined_snr = snr_db[:, reference_pos]
        relative_offsets = np.zeros(z.shape[1])
        relative_plv = np.zeros(z.shape[1])
        relative_plv[reference_pos] = 1.0
    elif mode == "mean":
        combined = np.nanmean(z, axis=1)
        combined_snr = np.nanmean(snr_db, axis=1)
        relative_offsets = np.zeros(z.shape[1])
        relative_plv = np.ones(z.shape[1])
    else:
        phase = np.angle(z)
        ref_phase = phase[:, reference_pos]
        relative_offsets = np.array([
            circ_mean(wrap_rad(phase[:, ch] - ref_phase)) for ch in range(z.shape[1])
        ])
        relative_plv = np.array([
            circ_r(wrap_rad(phase[:, ch] - ref_phase)) for ch in range(z.shape[1])
        ])
        relative_offsets[reference_pos] = 0.0
        relative_plv[reference_pos] = 1.0

        median_amp = np.nanmedian(np.abs(z), axis=0)
        median_amp = np.where(np.isfinite(median_amp) & (median_amp > 1e-12), median_amp, 1.0)
        corrected = z * np.exp(-1j * relative_offsets[None, :]) / median_amp[None, :]
        weights = np.where(np.isfinite(relative_plv), relative_plv, 0.0)
        if np.sum(weights) <= 0:
            weights = np.ones_like(weights)
        combined = np.nansum(corrected * weights[None, :], axis=1) / np.sum(weights)
        combined_snr = np.nansum(snr_db * weights[None, :], axis=1) / np.sum(weights)

    qc = {
        "relative_offset_rad": relative_offsets,
        "relative_plv": relative_plv,
    }
    return combined, combined_snr, qc


def flag_artifacts(trials: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = trials.copy()
    out["artifact_flag"] = False
    for (_, window), idx in out.groupby(["subject", "window_sec"], observed=True).groups.items():
        vals = out.loc[idx, "epoch_peak_to_peak"].to_numpy(dtype=float)
        med = np.nanmedian(vals)
        mad = robust_mad(vals)
        threshold = med + cfg.artifact_mad_threshold * mad if np.isfinite(mad) else float("inf")
        flags = vals > threshold
        if cfg.absolute_peak_to_peak_limit is not None:
            flags |= vals > cfg.absolute_peak_to_peak_limit
        out.loc[idx, "artifact_flag"] = flags
        out.loc[idx, "artifact_threshold"] = threshold
    out["included_primary"] = out["epoch_valid"].astype(bool)
    if cfg.reject_artifacts:
        out["included_primary"] &= ~out["artifact_flag"].astype(bool)
    return out


# =============================================================================
# 6. Statistical analyses
# =============================================================================


def fit_transfer_line(stim_rad: np.ndarray, response_rad: np.ndarray, theta_global: float) -> tuple[float, float, float]:
    unique = np.sort(np.unique(stim_rad))
    x = []
    y = []
    for phase in unique:
        mean_resp = circ_mean(response_rad[np.isclose(stim_rad, phase)])
        if np.isfinite(mean_resp):
            x.append(float(phase))
            y.append(float(phase + theta_global + wrap_rad(mean_resp - phase - theta_global)))
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    pred = design @ beta
    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[0]), float(beta[1]), r2


def permutation_transfer_test(
    response: np.ndarray,
    stim: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = circ_r(wrap_rad(response - stim))
    if iterations <= 0:
        return observed, float("nan")
    n = len(stim)
    perm_index = np.vstack([rng.permutation(n) for _ in range(iterations)])
    response_vector = np.exp(1j * response)[None, :]
    stim_vector = np.exp(-1j * stim[perm_index])
    perm = np.abs(np.mean(response_vector * stim_vector, axis=1))
    p = (1.0 + np.sum(perm >= observed)) / (iterations + 1.0)
    return observed, float(p)


def theta_constancy_test(
    delta: np.ndarray,
    phase_idx: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    global_theta = circ_mean(delta)

    def stat(labels: np.ndarray) -> float:
        values = []
        weights = []
        for label in np.unique(labels):
            group = delta[labels == label]
            if len(group):
                values.append(1.0 - math.cos(wrap_rad(circ_mean(group) - global_theta)))
                weights.append(len(group))
        return float(np.average(values, weights=weights)) if values else float("nan")

    observed = stat(phase_idx)
    if iterations <= 0 or not np.isfinite(observed):
        return observed, float("nan")
    perm = np.empty(iterations)
    for i in range(iterations):
        perm[i] = stat(rng.permutation(phase_idx))
    p = (1.0 + np.sum(perm >= observed)) / (iterations + 1.0)
    return observed, float(p)


def stratified_bootstrap_metrics(
    response: np.ndarray,
    stim: np.ndarray,
    phase_idx: np.ndarray,
    iterations: int,
    epsilon: float,
    max_m: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if iterations <= 0:
        return {}
    unique = np.unique(phase_idx)
    sigma_noise = np.empty(iterations)
    sigma_dec = np.empty(iterations)
    theta = np.empty(iterations)
    group_indices = {label: np.flatnonzero(phase_idx == label) for label in unique}
    for b in range(iterations):
        sampled = np.concatenate([
            rng.choice(group_indices[label], size=len(group_indices[label]), replace=True)
            for label in unique
        ])
        r = response[sampled]
        s = stim[sampled]
        p = phase_idx[sampled]
        d = wrap_rad(r - s)
        theta[b] = circ_mean(d)
        sigma_dec[b] = circ_std(wrap_rad(d - theta[b]))
        residual_noise = np.empty_like(d)
        for label in unique:
            mask = p == label
            th = circ_mean(d[mask])
            residual_noise[mask] = wrap_rad(d[mask] - th)
        sigma_noise[b] = circ_std(residual_noise)

    # Vectorized wrapped-normal Mmax calculation for all bootstrap samples.
    mmax = np.ones(iterations, dtype=float)
    sigma_safe = np.clip(sigma_dec, 1e-12, None)
    for m in range(2, max_m + 1):
        half_width = np.pi / m
        p_correct = np.zeros(iterations, dtype=float)
        for k in range(-12, 13):
            lower = (-half_width + 2.0 * np.pi * k) / sigma_safe
            upper = (half_width + 2.0 * np.pi * k) / sigma_safe
            p_correct += norm.cdf(upper) - norm.cdf(lower)
        valid = (1.0 - np.clip(p_correct, 0.0, 1.0)) <= epsilon
        mmax[valid] = m

    center_theta = circ_mean(theta)
    theta_unwrapped = center_theta + wrap_rad(theta - center_theta)
    return {
        "sigma_noise_ci_low": float(np.nanpercentile(sigma_noise, 2.5)),
        "sigma_noise_ci_high": float(np.nanpercentile(sigma_noise, 97.5)),
        "sigma_dec_ci_low": float(np.nanpercentile(sigma_dec, 2.5)),
        "sigma_dec_ci_high": float(np.nanpercentile(sigma_dec, 97.5)),
        "theta_ci_low_rad": float(np.nanpercentile(theta_unwrapped, 2.5)),
        "theta_ci_high_rad": float(np.nanpercentile(theta_unwrapped, 97.5)),
        "mmax_ci_low": float(np.nanpercentile(mmax, 2.5)),
        "mmax_ci_high": float(np.nanpercentile(mmax, 97.5)),
    }


def nearest_phase_predictions(
    response: np.ndarray,
    stim: np.ndarray,
    phase_idx: np.ndarray,
    phases_rad_by_idx: dict[int, float],
    calibrated: bool = False,
) -> np.ndarray:
    predictions = np.full(len(response), -1, dtype=int)
    labels = np.asarray(sorted(phases_rad_by_idx.keys()), dtype=int)
    for test in range(len(response)):
        train = np.arange(len(response)) != test
        delta_train = wrap_rad(response[train] - stim[train])
        theta_global = circ_mean(delta_train)
        centers = []
        for label in labels:
            if calibrated:
                mask = train & (phase_idx == label)
                theta_label = circ_mean(wrap_rad(response[mask] - stim[mask])) if np.any(mask) else theta_global
            else:
                theta_label = theta_global
            centers.append(wrap_rad(phases_rad_by_idx[int(label)] + theta_label))
        distances = circ_distance(response[test], np.asarray(centers))
        predictions[test] = int(labels[int(np.argmin(distances))])
    return predictions


def confusion_counts(true: np.ndarray, pred: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    labels = list(labels)
    lookup = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        if int(t) in lookup and int(p) in lookup:
            cm[lookup[int(t)], lookup[int(p)]] += 1
    return cm


def residual_distribution_metrics(residual: np.ndarray, sigma: float) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    out: dict[str, float] = {
        "residual_skew": float(skew(residual, bias=False)) if len(residual) >= 3 else float("nan"),
        "residual_excess_kurtosis": float(kurtosis(residual, fisher=True, bias=False)) if len(residual) >= 4 else float("nan"),
    }
    if 3 <= len(residual) <= 5000:
        try:
            sh = shapiro(residual)
            out["shapiro_w"] = float(sh.statistic)
            out["shapiro_p"] = float(sh.pvalue)
        except Exception:
            out["shapiro_w"] = float("nan")
            out["shapiro_p"] = float("nan")
    else:
        out["shapiro_w"] = float("nan")
        out["shapiro_p"] = float("nan")

    try:
        kappa, loc, _ = vonmises.fit(residual, fscale=1.0)
        vm_ll = float(np.sum(vonmises.logpdf(residual, kappa, loc=loc)))
    except Exception:
        kappa, loc, vm_ll = float("nan"), float("nan"), float("nan")
    wn_pdf = wrapped_normal_pdf(residual, sigma)
    wn_ll = float(np.sum(np.log(np.clip(wn_pdf, 1e-300, None)))) if np.all(np.isfinite(wn_pdf)) else float("nan")
    out.update({
        "vonmises_kappa": float(kappa),
        "vonmises_loc_rad": float(loc),
        "aic_vonmises": float(2 * 2 - 2 * vm_ll) if np.isfinite(vm_ll) else float("nan"),
        "aic_wrapped_normal": float(2 * 1 - 2 * wn_ll) if np.isfinite(wn_ll) else float("nan"),
    })
    return out


def analyze_subject_window(
    data: pd.DataFrame,
    cfg: AnalysisConfig,
    seed_offset: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    use = data[data["included_primary"]].copy()
    use = use[np.isfinite(use["response_phase_rad"]) & np.isfinite(use["stim_phase_rad"])]
    if use.empty:
        raise ValueError("No valid trials in this subject/window")

    response = use["response_phase_rad"].to_numpy(dtype=float)
    stim = use["stim_phase_rad"].to_numpy(dtype=float)
    phase_idx = use["phase_idx"].to_numpy(dtype=int)
    delta = wrap_rad(response - stim)
    theta_global = circ_mean(delta)
    residual_dec = wrap_rad(delta - theta_global)

    phase_rows = []
    residual_noise = np.empty_like(delta)
    theta_by_phase = []
    for label in sorted(np.unique(phase_idx)):
        mask = phase_idx == label
        theta_m = circ_mean(delta[mask])
        theta_by_phase.append(theta_m)
        residual_noise[mask] = wrap_rad(delta[mask] - theta_m)
        phase_rows.append({
            "phase_idx": int(label),
            "phase_deg": float(np.rad2deg(circ_mean(stim[mask])) % 360.0),
            "n_trials": int(np.sum(mask)),
            "response_mean_rad": circ_mean(response[mask]),
            "response_mean_deg": float(np.rad2deg(circ_mean(response[mask])) % 360.0),
            "theta_m_rad": theta_m,
            "theta_m_deg": float(np.rad2deg(theta_m)),
            "phase_plv": circ_r(response[mask]),
            "sigma_noise_phase_rad": circ_std(residual_noise[mask]),
            "sigma_noise_phase_deg": float(np.rad2deg(circ_std(residual_noise[mask]))),
            "mean_snr_db": float(np.nanmean(use.loc[mask, "snr_db"])),
        })

    sigma_noise = circ_std(residual_noise)
    sigma_dec = circ_std(residual_dec)
    transfer_plv = circ_r(delta)
    slope, intercept, transfer_r2 = fit_transfer_line(stim, response, theta_global)
    theta_range = circular_peak_to_peak(theta_by_phase, theta_global)

    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    transfer_stat, transfer_p = permutation_transfer_test(
        response, stim, cfg.permutation_iterations, rng
    )
    theta_stat, theta_p = theta_constancy_test(
        delta, phase_idx, cfg.permutation_iterations, rng
    )
    boot = stratified_bootstrap_metrics(
        response,
        stim,
        phase_idx,
        cfg.bootstrap_iterations,
        cfg.capacity_error_threshold,
        cfg.capacity_max_m,
        rng,
    )

    # Variance homogeneity on per-phase-centered signed residuals.
    groups = [residual_noise[phase_idx == label] for label in sorted(np.unique(phase_idx))]
    try:
        lev = levene(*groups, center="median") if all(len(g) >= 2 for g in groups) else None
        levene_stat = float(lev.statistic) if lev is not None else float("nan")
        levene_p = float(lev.pvalue) if lev is not None else float("nan")
    except Exception:
        levene_stat, levene_p = float("nan"), float("nan")

    phases_rad_by_idx = {
        int(label): circ_mean(stim[phase_idx == label]) for label in sorted(np.unique(phase_idx))
    }
    pred_global = nearest_phase_predictions(response, stim, phase_idx, phases_rad_by_idx, calibrated=False)
    pred_cal = nearest_phase_predictions(response, stim, phase_idx, phases_rad_by_idx, calibrated=True)
    errors = pred_global != phase_idx
    errors_cal = pred_cal != phase_idx
    n = len(use)
    k = int(np.sum(errors))
    k_cal = int(np.sum(errors_cal))
    error_rate = k / n
    error_rate_cal = k_cal / n
    ci_low, ci_high = wilson_interval(k, n, cfg.alpha)

    m_actual = len(phases_rad_by_idx)
    label_order = sorted(phases_rad_by_idx)
    cm = confusion_counts(phase_idx, pred_global, label_order)
    cm_cal = confusion_counts(phase_idx, pred_cal, label_order)
    if k > 0:
        circular_steps = np.minimum(
            np.abs(pred_global[errors] - phase_idx[errors]),
            m_actual - np.abs(pred_global[errors] - phase_idx[errors]),
        )
        adjacent_error_fraction = float(np.mean(circular_steps == 1))
    else:
        adjacent_error_fraction = float("nan")

    predicted_gaussian = predicted_error_gaussian(m_actual, sigma_dec)
    predicted_wn = predicted_error_wrapped_normal(m_actual, sigma_dec)
    mmax_gaussian = mmax_closed_form(sigma_dec, cfg.capacity_error_threshold)
    mmax_wn = mmax_wrapped_normal(sigma_dec, cfg.capacity_error_threshold, cfg.capacity_max_m)

    dist = residual_distribution_metrics(residual_dec, sigma_dec)
    outlier_fraction = float(np.mean(np.abs(residual_dec) > 3.0 * sigma_dec)) if sigma_dec > 0 else 0.0

    min_per_phase = int(use.groupby("phase_idx", observed=True).size().min())
    transfer_support = bool(
        np.isfinite(transfer_p)
        and transfer_p < cfg.alpha
        and np.isfinite(slope)
        and abs(slope - 1.0) <= cfg.slope_tolerance
    )
    theta_stable = bool(
        (not np.isfinite(theta_p) or theta_p >= cfg.alpha)
        and np.rad2deg(theta_range) <= cfg.theta_range_tolerance_deg
    )
    locking_adequate = bool(np.rad2deg(sigma_dec) < cfg.weak_locking_sigma_deg)
    model_prediction_consistent = bool(ci_low <= predicted_wn <= ci_high)

    summary = {
        "subject": str(use["subject"].iloc[0]),
        "stim_freq_hz": float(use["stim_freq_hz"].iloc[0]),
        "window_sec": float(use["window_sec"].iloc[0]),
        "transient_sec": float(use["transient_sec"].iloc[0]),
        "n_trials": n,
        "n_phases": m_actual,
        "min_trials_per_phase": min_per_phase,
        "confirmatory_sample_size": min_per_phase >= cfg.min_trials_per_phase_for_confirmatory,
        "theta_global_rad": theta_global,
        "theta_global_deg": float(np.rad2deg(theta_global)),
        "theta_range_rad": theta_range,
        "theta_range_deg": float(np.rad2deg(theta_range)),
        "sigma_noise_rad": sigma_noise,
        "sigma_noise_deg": float(np.rad2deg(sigma_noise)),
        "sigma_dec_rad": sigma_dec,
        "sigma_dec_deg": float(np.rad2deg(sigma_dec)),
        "transfer_plv": transfer_plv,
        "transfer_permutation_stat": transfer_stat,
        "transfer_permutation_p": transfer_p,
        "theta_constancy_stat": theta_stat,
        "theta_constancy_p": theta_p,
        "transfer_slope": slope,
        "transfer_intercept_rad": intercept,
        "transfer_r2": transfer_r2,
        "levene_stat": levene_stat,
        "levene_p": levene_p,
        "mean_snr_db": float(np.nanmean(use["snr_db"])),
        "median_snr_db": float(np.nanmedian(use["snr_db"])),
        "artifact_fraction": float(np.mean(data["artifact_flag"])),
        "empirical_error_rate": error_rate,
        "empirical_error_ci_low": ci_low,
        "empirical_error_ci_high": ci_high,
        "calibrated_error_rate": error_rate_cal,
        "adjacent_error_fraction": adjacent_error_fraction,
        "predicted_error_gaussian": predicted_gaussian,
        "predicted_error_wrapped_normal": predicted_wn,
        "mmax_closed_form": mmax_gaussian,
        "mmax_wrapped_normal": mmax_wn,
        "outlier_3sigma_fraction": outlier_fraction,
        "transfer_support": transfer_support,
        "theta_stable": theta_stable,
        "locking_adequate": locking_adequate,
        "prediction_consistent_with_wilson_ci": model_prediction_consistent,
        **dist,
        **boot,
    }

    phase_df = pd.DataFrame(phase_rows)
    phase_df.insert(0, "window_sec", summary["window_sec"])
    phase_df.insert(0, "subject", summary["subject"])

    details = {
        "used_trials": use,
        "response": response,
        "stim": stim,
        "phase_idx": phase_idx,
        "delta": delta,
        "residual_dec": residual_dec,
        "residual_noise": residual_noise,
        "pred_global": pred_global,
        "pred_calibrated": pred_cal,
        "cm": cm,
        "cm_calibrated": cm_cal,
        "label_order": label_order,
    }
    return summary, phase_df, details


def fit_window_effect(window_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, df in window_summary.groupby("subject", observed=True):
        d = df.sort_values("window_sec")
        valid = (d["sigma_noise_rad"] > 0) & np.isfinite(d["sigma_noise_rad"]) & (d["window_sec"] > 0)
        if valid.sum() >= 2:
            x = np.log(d.loc[valid, "window_sec"].to_numpy(dtype=float))
            y = np.log(d.loc[valid, "sigma_noise_rad"].to_numpy(dtype=float))
            design = np.column_stack([x, np.ones_like(x)])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred = design @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            slope = float(beta[0])
        else:
            slope, r2 = np.nan, np.nan
        rows.append({
            "subject": subject,
            "loglog_slope_sigma_noise_vs_T": slope,
            "loglog_r2": float(r2),
            "decreases_with_window": bool(np.isfinite(slope) and slope < 0),
            "approximately_inverse_sqrt": bool(np.isfinite(slope) and abs(slope + 0.5) <= 0.25),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. Plotting
# =============================================================================


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transfer(details: dict[str, Any], summary: dict[str, Any], path: Path, dpi: int) -> None:
    stim = details["stim"]
    response = details["response"]
    theta = summary["theta_global_rad"]
    x_deg = np.rad2deg(stim)
    y_unwrapped = stim + theta + wrap_rad(response - stim - theta)
    y_deg = np.rad2deg(y_unwrapped)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x_deg, y_deg, alpha=0.55, label="Trial")
    unique = np.sort(np.unique(stim))
    mean_y = []
    sd_y = []
    for p in unique:
        mask = np.isclose(stim, p)
        mean_resp = circ_mean(response[mask])
        center = p + theta + wrap_rad(mean_resp - p - theta)
        residual = wrap_rad(response[mask] - mean_resp)
        mean_y.append(np.rad2deg(center))
        sd_y.append(np.rad2deg(circ_std(residual)))
    ax.errorbar(np.rad2deg(unique), mean_y, yerr=sd_y, marker="o", linestyle="none", capsize=3, label="Circular mean ± circular SD")
    xx = np.linspace(np.min(x_deg), np.max(x_deg), 200)
    ax.plot(xx, xx + np.rad2deg(theta), linestyle="--", label="Slope 1 + global theta")
    ax.plot(xx, summary["transfer_slope"] * np.deg2rad(xx) * 180.0 / np.pi + np.rad2deg(summary["transfer_intercept_rad"]), label="Fitted class-mean line")
    ax.set_xlabel("Stimulus phase (deg)")
    ax.set_ylabel("Unwrapped response phase (deg)")
    ax.set_title(f"{summary['subject']} phase transfer, T={summary['window_sec']:.2f} s")
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_figure(fig, path, dpi)


def plot_theta_by_phase(phase_df: pd.DataFrame, summary: dict[str, Any], path: Path, dpi: int) -> None:
    p = phase_df.sort_values("phase_deg")
    theta_deg = np.rad2deg(
        summary["theta_global_rad"] + wrap_rad(p["theta_m_rad"].to_numpy() - summary["theta_global_rad"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(p["phase_deg"], theta_deg, marker="o", label="Phase-specific theta")
    ax.axhline(summary["theta_global_deg"], linestyle="--", label="Global theta")
    ax.set_xlabel("Stimulus phase (deg)")
    ax.set_ylabel("Transmission shift theta (deg, unwrapped around global mean)")
    ax.set_title(
        f"Theta constancy: range={summary['theta_range_deg']:.1f}°, p={summary['theta_constancy_p']:.3g}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path, dpi)


def plot_residuals(details: dict[str, Any], summary: dict[str, Any], path: Path, dpi: int) -> None:
    residual = details["residual_dec"]
    sigma = summary["sigma_dec_rad"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].hist(np.rad2deg(residual), bins="auto", density=True, alpha=0.65, label="Residuals")
    grid = np.linspace(-np.pi, np.pi, 600)
    axes[0].plot(np.rad2deg(grid), wrapped_normal_pdf(grid, sigma) * np.pi / 180.0, label="Wrapped normal")
    if np.isfinite(summary.get("vonmises_kappa", np.nan)):
        vm = vonmises.pdf(grid, summary["vonmises_kappa"], loc=summary["vonmises_loc_rad"])
        axes[0].plot(np.rad2deg(grid), vm * np.pi / 180.0, label="von Mises")
    axes[0].axvline(0.0, linestyle="--")
    axes[0].set_xlabel("Global-theta residual (deg)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"sigma_dec={summary['sigma_dec_deg']:.1f}°")
    axes[0].legend()

    (osm, osr), (slope, intercept, r) = probplot(residual, dist="norm")
    axes[1].scatter(osm, osr, alpha=0.7)
    axes[1].plot(osm, slope * np.asarray(osm) + intercept, linestyle="--")
    axes[1].set_xlabel("Theoretical normal quantiles")
    axes[1].set_ylabel("Ordered residuals (rad)")
    axes[1].set_title(f"Normal Q-Q, Shapiro p={summary.get('shapiro_p', np.nan):.3g}")
    axes[1].grid(True, alpha=0.25)
    save_figure(fig, path, dpi)


def plot_confusion(details: dict[str, Any], phase_df: pd.DataFrame, summary: dict[str, Any], path: Path, dpi: int) -> None:
    cm = details["cm"]
    labels = details["label_order"]
    phase_lookup = phase_df.set_index("phase_idx")["phase_deg"].to_dict()
    tick = [f"{phase_lookup.get(label, label):.0f}°" for label in labels]
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, label="Row-normalized proportion")
    ax.set_xticks(range(len(tick)), tick)
    ax.set_yticks(range(len(tick)), tick)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_title(
        f"LOO global-theta decoding, error={summary['empirical_error_rate']:.1%}\n"
        f"Adjacent fraction among errors={summary['adjacent_error_fraction']:.1%}"
        if np.isfinite(summary["adjacent_error_fraction"])
        else f"LOO global-theta decoding, error={summary['empirical_error_rate']:.1%}"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center")
    save_figure(fig, path, dpi)


def plot_capacity(summary: dict[str, Any], path: Path, dpi: int, cfg: AnalysisConfig) -> None:
    m_values = np.arange(2, cfg.capacity_max_m + 1)
    sigma = summary["sigma_dec_rad"]
    pg = np.array([predicted_error_gaussian(int(m), sigma) for m in m_values])
    pw = np.array([predicted_error_wrapped_normal(int(m), sigma) for m in m_values])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(m_values, pg, marker="o", markersize=3, label="2Q Gaussian approximation")
    ax.plot(m_values, pw, marker="s", markersize=3, label="Wrapped-normal decision integral")
    ax.axhline(cfg.capacity_error_threshold, linestyle="--", label=f"Error threshold={cfg.capacity_error_threshold:.0%}")
    ax.errorbar(
        [summary["n_phases"]],
        [summary["empirical_error_rate"]],
        yerr=[
            [summary["empirical_error_rate"] - summary["empirical_error_ci_low"]],
            [summary["empirical_error_ci_high"] - summary["empirical_error_rate"]],
        ],
        marker="D",
        capsize=4,
        linestyle="none",
        label="Empirical Stage-1 M with Wilson CI",
    )
    ax.set_xlabel("Number of phase classes M")
    ax.set_ylabel("Error probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Capacity prediction from sigma_dec={summary['sigma_dec_deg']:.1f}°; "
        f"Mmax={summary['mmax_wrapped_normal']}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path, dpi)


def plot_polar(details: dict[str, Any], phase_df: pd.DataFrame, summary: dict[str, Any], path: Path, dpi: int) -> None:
    response = details["response"]
    phase_idx = details["phase_idx"]
    fig = plt.figure(figsize=(7.0, 6.3))
    ax = fig.add_subplot(111, projection="polar")
    for label in sorted(np.unique(phase_idx)):
        mask = phase_idx == label
        radius = 1.0 + 0.035 * np.arange(np.sum(mask))
        phase_deg = phase_df.loc[phase_df["phase_idx"] == label, "phase_deg"].iloc[0]
        ax.scatter(response[mask], radius, s=22, alpha=0.65, label=f"Stim {phase_deg:.0f}°")
        mean_resp = circ_mean(response[mask])
        ax.plot([mean_resp, mean_resp], [0, 1.3], linewidth=2)
    ax.set_title(f"Response-phase polar scatter, {summary['subject']}, T={summary['window_sec']:.2f} s")
    ax.set_yticklabels([])
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    save_figure(fig, path, dpi)


def plot_window_effect(subject_df: pd.DataFrame, trend_row: pd.Series, path: Path, dpi: int) -> None:
    d = subject_df.sort_values("window_sec")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(
        d["window_sec"],
        d["sigma_noise_deg"],
        yerr=[
            d["sigma_noise_deg"] - np.rad2deg(d.get("sigma_noise_ci_low", d["sigma_noise_rad"])),
            np.rad2deg(d.get("sigma_noise_ci_high", d["sigma_noise_rad"])) - d["sigma_noise_deg"],
        ],
        marker="o",
        capsize=3,
        label="sigma_noise",
    )
    ax.plot(d["window_sec"], d["sigma_dec_deg"], marker="s", label="sigma_dec")
    if len(d) >= 2 and d["sigma_noise_deg"].iloc[0] > 0:
        ref = d["sigma_noise_deg"].iloc[0] * np.sqrt(d["window_sec"].iloc[0] / d["window_sec"])
        ax.plot(d["window_sec"], ref, linestyle="--", label="1/sqrt(T) anchored at shortest window")
    ax.set_xlabel("Analysis window T (s)")
    ax.set_ylabel("Circular SD (deg)")
    ax.set_title(
        f"{d['subject'].iloc[0]} window effect; log-log slope="
        f"{trend_row.get('loglog_slope_sigma_noise_vs_T', np.nan):.2f}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path, dpi)


def plot_group_window_effect(summary_df: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for subject, d in summary_df.groupby("subject", observed=True):
        d = d.sort_values("window_sec")
        ax.plot(d["window_sec"], d["sigma_noise_deg"], marker="o", alpha=0.45, label=str(subject))
    group = summary_df.groupby("window_sec", observed=True)["sigma_noise_deg"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(group["window_sec"], group["mean"], yerr=group["sem"].fillna(0), marker="D", linewidth=2.2, capsize=4, label="Group mean ± SEM")
    ax.set_xlabel("Analysis window T (s)")
    ax.set_ylabel("sigma_noise (deg)")
    ax.set_title("Window-length effect across subjects")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    save_figure(fig, path, dpi)


def plot_group_mmax(summary_df: pd.DataFrame, classification_window: float, path: Path, dpi: int) -> None:
    available = np.sort(summary_df["window_sec"].unique())
    chosen = float(available[np.argmin(np.abs(available - classification_window))])
    d = summary_df[np.isclose(summary_df["window_sec"], chosen)].sort_values("subject")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(d) + 3), 4.8))
    ax.bar(d["subject"], d["mmax_wrapped_normal"])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Predicted Mmax")
    ax.set_title(f"Predicted phase capacity at T={chosen:.2f} s")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, path, dpi)


# =============================================================================
# 8. End-to-end pipeline
# =============================================================================


def process_one_file(
    subject: str,
    file_path: Path,
    file_index: int,
    cfg: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mat = load_mat_file(file_path)
    if "data" not in mat:
        raise KeyError(f"{file_path} does not contain variable 'data'")
    data = orient_data_matrix(mat["data"])

    # First parse schedule without needing marker events.
    schedule = parse_matrix_with_columns(mat, "trial_schedule", "trial_schedule_columns")

    # Infer phase/trigger values enough to read the fixed marker channel.
    phase_count = cfg.phase_count_override or _safe_int(mat.get("phase_count"), None)
    trigger_base = cfg.trigger_base_override or _safe_int(mat.get("trigger_base"), 1) or 1
    if not schedule.empty and "trigger" in schedule.columns:
        expected_labels = sorted(schedule["trigger"].astype(int).unique())
    elif phase_count is not None:
        expected_labels = list(range(trigger_base, trigger_base + phase_count))
    else:
        expected_labels = list(range(trigger_base, trigger_base + 16))

    fs_pre = cfg.fs_override if cfg.fs_override is not None else _safe_float(mat.get("Fs"))
    if not np.isfinite(fs_pre) or fs_pre <= 0:
        raise ValueError(f"Invalid Fs in {file_path}; use --fs-override")

    marker_channel, events, marker_qc = resolve_marker_channel(data, expected_labels, fs_pre, cfg)
    expected = build_expected_trials(mat, cfg, events)
    params = infer_experiment_params(mat, expected, cfg)
    fs = params["fs"]

    # Re-extract events after exact trigger labels are known.
    expected_labels = sorted(expected["trigger"].astype(int).unique())
    events = extract_marker_events(
        data[:, marker_channel], expected_labels, fs, cfg.marker_integer_tolerance, cfg.min_inter_event_sec
    )
    aligned, alignment_qc = align_trials_to_samples(expected, events, fs, cfg)
    aligned = apply_trial_subset(aligned, cfg)

    match_fraction = aligned["onset_sample"].notna().mean() if len(aligned) else 0.0
    if match_fraction < cfg.min_event_match_fraction:
        raise RuntimeError(
            f"Low event-match fraction in {file_path}: {match_fraction:.1%}. "
            f"Detected marker channel={marker_channel}, events={len(events)}, expected={len(expected)}. "
            "Check the fixed marker channel, set --marker-channel explicitly, or enable "
            "--allow-timing-fallback only after verifying the timing offset."
        )

    eeg_indices, channel_names, reference_pos = resolve_channel_indices(data.shape[1], marker_channel, cfg)
    eeg = preprocess_continuous(extract_eeg_matrix(data, eeg_indices, cfg), fs, cfg)

    trial_records: list[dict[str, Any]] = []
    channel_records: list[dict[str, Any]] = []
    alignment_rows = aligned.copy()
    alignment_rows.insert(0, "file", str(file_path))
    alignment_rows.insert(0, "session", file_path.stem)
    alignment_rows.insert(0, "subject", subject)

    for window_sec in cfg.window_lengths_sec:
        z_rows = []
        snr_rows = []
        meta_rows = []
        p2p_rows = []
        n_samples = int(round(window_sec * fs))
        transient_samples = int(round(cfg.transient_sec * fs))

        for row_idx, trial in aligned.iterrows():
            if not np.isfinite(trial["onset_sample"]):
                continue
            onset = int(round(trial["onset_sample"]))
            start = onset + transient_samples
            stop = start + n_samples
            epoch_valid = start >= 0 and stop <= len(eeg)
            if np.isfinite(params["stimulus_duration_sec"]):
                epoch_valid &= cfg.transient_sec + window_sec <= params["stimulus_duration_sec"] + 1e-9
            if not epoch_valid:
                continue
            epoch = eeg[start:stop, :]
            z = harmonic_regression_coefficients(epoch, fs, params["stim_freq"])
            snr = regression_snr_db(epoch, fs, params["stim_freq"])
            p2p = float(np.nanmax(np.ptp(epoch, axis=0)))
            z_rows.append(z)
            snr_rows.append(snr)
            p2p_rows.append(p2p)
            meta_rows.append((row_idx, trial))

        if not z_rows:
            continue
        z_arr = np.vstack(z_rows)
        snr_arr = np.vstack(snr_rows)
        combined, combined_snr, spatial_qc = combine_channel_coefficients(
            z_arr, snr_arr, cfg.spatial_mode, reference_pos
        )

        for local_i, (row_idx, trial) in enumerate(meta_rows):
            phase_deg = float(trial.get("phase_deg", np.nan))
            if not np.isfinite(phase_deg):
                phase_idx = int(trial["phase_idx"])
                phase_deg = float(params["phases_deg"][phase_idx])
            record = {
                "subject": subject,
                "session": file_path.stem,
                "file": str(file_path),
                "file_index": file_index,
                "trial_idx": int(trial["trial_idx"]),
                "block_idx": int(trial.get("block_idx", 0)),
                "trial_in_block": int(trial.get("trial_in_block", trial["trial_idx"])),
                "phase_idx": int(trial["phase_idx"]),
                "phase_deg": phase_deg,
                "stim_phase_rad": float(np.deg2rad(phase_deg)),
                "trigger": int(trial["trigger"]),
                "onset_sample": int(round(trial["onset_sample"])),
                "alignment_source": str(trial["alignment_source"]),
                "fs_hz": fs,
                "stim_freq_hz": params["stim_freq"],
                "transient_sec": cfg.transient_sec,
                "window_sec": float(window_sec),
                "response_phase_rad": float(np.angle(combined[local_i])),
                "response_phase_deg": float(np.rad2deg(np.angle(combined[local_i])) % 360.0),
                "response_amplitude": float(np.abs(combined[local_i])),
                "snr_db": float(combined_snr[local_i]),
                "epoch_peak_to_peak": p2p_rows[local_i],
                "epoch_valid": True,
                "marker_channel": marker_channel,
                "reference_eeg_channel": eeg_indices[reference_pos],
                "spatial_mode": cfg.spatial_mode,
                "linked_mastoids_reference": cfg.linked_mastoids_reference,
            }
            trial_records.append(record)

            if cfg.save_trial_channel_metrics:
                for pos, abs_ch in enumerate(eeg_indices):
                    channel_records.append({
                        "subject": subject,
                        "session": file_path.stem,
                        "file": str(file_path),
                        "trial_idx": int(trial["trial_idx"]),
                        "window_sec": float(window_sec),
                        "channel_index": abs_ch,
                        "channel_name": channel_names[pos],
                        "phase_rad": float(np.angle(z_arr[local_i, pos])),
                        "amplitude": float(np.abs(z_arr[local_i, pos])),
                        "snr_db": float(snr_arr[local_i, pos]),
                        "relative_offset_rad": float(spatial_qc["relative_offset_rad"][pos]),
                        "relative_plv": float(spatial_qc["relative_plv"][pos]),
                    })

    marker_qc = marker_qc.copy()
    marker_qc.insert(0, "file", str(file_path))
    marker_qc.insert(0, "session", file_path.stem)
    marker_qc.insert(0, "subject", subject)

    file_qc = {
        "subject": subject,
        "session": file_path.stem,
        "file": str(file_path),
        "n_samples": int(data.shape[0]),
        "n_columns": int(data.shape[1]),
        "fs_hz": fs,
        "stim_freq_hz": params["stim_freq"],
        "phase_count": params["phase_count"],
        "phases_deg": ",".join(f"{v:g}" for v in params["phases_deg"]),
        "marker_channel": marker_channel,
        "eeg_channels": ",".join(map(str, eeg_indices)),
        "reference_eeg_channel": eeg_indices[reference_pos],
        "linked_mastoids_reference": cfg.linked_mastoids_reference,
        **alignment_qc,
        "retained_trials_after_subset": int(aligned["onset_sample"].notna().sum()),
        "dropped_frames_metadata": _safe_int(mat.get("dropped_frames"), None),
        "measured_fps_metadata": _safe_float(mat.get("measured_fps")),
    }

    return (
        pd.DataFrame(trial_records),
        pd.DataFrame(channel_records),
        marker_qc,
        {"file_qc": file_qc, "alignment": alignment_rows},
    )


def select_classification_window(summary_df: pd.DataFrame, requested: float) -> float:
    available = np.sort(summary_df["window_sec"].unique())
    return float(available[np.argmin(np.abs(available - requested))])


def write_markdown_report(
    output_path: Path,
    cfg: AnalysisConfig,
    file_qc_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> None:
    selected_window = select_classification_window(summary_df, cfg.classification_window_sec)
    selected = summary_df[np.isclose(summary_df["window_sec"], selected_window)].copy()
    trend_lookup = trend_df.set_index("subject").to_dict(orient="index") if not trend_df.empty else {}

    lines = [
        "# Stage 1 同频相位传递分析报告",
        "",
        "## 数据与对齐质控",
        "",
        f"- 发现被试数：{summary_df['subject'].nunique()}",
        f"- 分析文件数：{len(file_qc_df)}",
        f"- 主分类/容量展示窗长：{selected_window:.2f} s（请求值 {cfg.classification_window_sec:.2f} s）",
        f"- 起始瞬态丢弃：{cfg.transient_sec:.2f} s",
        f"- 主分析是否剔除伪迹 trial：{'是' if cfg.reject_artifacts else '否'}",
        "",
    ]

    for _, row in file_qc_df.iterrows():
        lines.append(
            f"- `{row['subject']}` / `{row['session']}`：marker 列 {row['marker_channel']}，"
            f"检测 {row['detected_event_count']} 个事件，匹配 {row['matched_trial_count']}/"
            f"{row['expected_trial_count']}（{row['match_fraction']:.1%}），方法 `{row['alignment_method']}`。"
        )

    lines += [
        "",
        "## 主要结果",
        "",
        "下表中的自动判断只用于快速筛查，阈值均可在 `AnalysisConfig` 中修改；正式结论应结合图形、置信区间和实验重复。",
        "",
        "| 被试 | trials/相位 | slope | transfer p | theta范围 | sigma_noise | sigma_dec | M=实际经验错误 | 预测错误 | 预测Mmax | 初步判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in selected.sort_values("subject").iterrows():
        judgments = []
        judgments.append("相位传递支持" if row["transfer_support"] else "相位传递证据不足/偏离")
        judgments.append("theta稳定" if row["theta_stable"] else "theta存在相位依赖")
        judgments.append("锁相尚可" if row["locking_adequate"] else "锁相偏弱")
        judgments.append("预测落入CI" if row["prediction_consistent_with_wilson_ci"] else "预测偏离经验CI")
        if not row["confirmatory_sample_size"]:
            judgments.append("样本量仅探索性")
        lines.append(
            f"| {row['subject']} | {int(row['min_trials_per_phase'])} | {row['transfer_slope']:.3f} | "
            f"{row['transfer_permutation_p']:.3g} | {row['theta_range_deg']:.1f}° | "
            f"{row['sigma_noise_deg']:.1f}° | {row['sigma_dec_deg']:.1f}° | "
            f"{row['empirical_error_rate']:.1%} [{row['empirical_error_ci_low']:.1%}, {row['empirical_error_ci_high']:.1%}] | "
            f"{row['predicted_error_wrapped_normal']:.1%} | {int(row['mmax_wrapped_normal'])} | "
            f"{'；'.join(judgments)} |"
        )

    lines += [
        "",
        "## 窗长效应",
        "",
    ]
    for _, row in trend_df.sort_values("subject").iterrows():
        if np.isfinite(row["loglog_slope_sigma_noise_vs_T"]):
            lines.append(
                f"- {row['subject']}：log(sigma_noise)-log(T) 斜率 "
                f"{row['loglog_slope_sigma_noise_vs_T']:.3f}，R²={row['loglog_r2']:.3f}。"
                f"{'接近 -0.5。' if row['approximately_inverse_sqrt'] else '未达到预设的 -0.5±0.25 范围。'}"
            )

    lines += [
        "",
        "## 解释顺序",
        "",
        "1. 先看 `transfer` 与 `theta_by_phase` 图：判断响应相位是否保持顺序、斜率是否接近 1、各相位的 theta 是否近似恒定。",
        "2. 再看 `residuals`：比较 sigma_noise 与 sigma_dec，检查单峰性、重尾和分相位方差差异。",
        "3. 看 `window_effect`：若 sigma_noise 随 T 增大下降且 log-log 斜率接近 -0.5，支持估计噪声随窗长下降；若趋于平台，提示生理抖动或系统时序误差占主导。",
        "4. 看 `confusion`：错误应主要落在相邻相位；大量非相邻错误会违背最近邻 M-PSK 假设。",
        "5. 看 `capacity`：Stage 1 只有当前 M 的经验点，其他 M 只是由 sigma_dec 外推，必须由 Stage 2 不同 M 的独立数据验证。",
        "",
        "## 样本量提醒",
        "",
        f"当前配置把每相位少于 {cfg.min_trials_per_phase_for_confirmatory} 个有效 trial 标为探索性。"
        "每相位 10 个 trial 可以用于流程调试和初步判断，但对残差分布、方差齐性、10% 错误率边界和 Mmax 的置信区间仍然较宽。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(cfg: AnalysisConfig) -> None:
    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "figures").mkdir(exist_ok=True)
    (cfg.output_dir / "tables").mkdir(exist_ok=True)

    config_path = cfg.output_dir / "analysis_config.json"
    config_path.write_text(json.dumps(to_jsonable(asdict(cfg)), ensure_ascii=False, indent=2), encoding="utf-8")

    subject_files = discover_subject_files(cfg)
    print(f"[Stage1 analysis] Found {len(subject_files)} subject(s).")

    trial_tables = []
    channel_tables = []
    marker_tables = []
    file_qc_rows = []
    alignment_tables = []

    for subject, files in subject_files.items():
        print(f"[Stage1 analysis] {subject}: {len(files)} file(s)")
        for file_index, path in enumerate(files):
            print(f"  -> {path}")
            trials, channels, marker_qc, extra = process_one_file(subject, path, file_index, cfg)
            trial_tables.append(trials)
            if not channels.empty:
                channel_tables.append(channels)
            marker_tables.append(marker_qc)
            file_qc_rows.append(extra["file_qc"])
            alignment_tables.append(extra["alignment"])

    trial_df = pd.concat(trial_tables, ignore_index=True)
    trial_df = flag_artifacts(trial_df, cfg)
    channel_df = pd.concat(channel_tables, ignore_index=True) if channel_tables else pd.DataFrame()
    marker_df = pd.concat(marker_tables, ignore_index=True)
    file_qc_df = pd.DataFrame(file_qc_rows)
    alignment_df = pd.concat(alignment_tables, ignore_index=True)

    table_dir = cfg.output_dir / "tables"
    trial_df.to_csv(table_dir / "trial_level_phase_estimates.csv", index=False, encoding="utf-8-sig")
    if not channel_df.empty:
        channel_df.to_csv(table_dir / "trial_channel_metrics.csv", index=False, encoding="utf-8-sig")
    marker_df.to_csv(table_dir / "marker_channel_qc.csv", index=False, encoding="utf-8-sig")
    file_qc_df.to_csv(table_dir / "file_qc.csv", index=False, encoding="utf-8-sig")
    alignment_df.to_csv(table_dir / "trial_alignment.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    phase_tables = []
    details_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    group_cols = ["subject", "window_sec"]
    for seed_offset, ((subject, window), d) in enumerate(trial_df.groupby(group_cols, observed=True)):
        try:
            summary, phase_df, details = analyze_subject_window(d, cfg, seed_offset * 10007)
        except ValueError as exc:
            warnings.warn(f"Skipping {subject}, T={window}: {exc}")
            continue
        summary_rows.append(summary)
        phase_tables.append(phase_df)
        details_by_key[(str(subject), float(window))] = details

    if not summary_rows:
        raise RuntimeError("No subject/window produced valid analysis results")

    summary_df = pd.DataFrame(summary_rows).sort_values(["subject", "window_sec"]).reset_index(drop=True)
    phase_df_all = pd.concat(phase_tables, ignore_index=True)
    trend_df = fit_window_effect(summary_df)

    # Descriptive per-file/session summaries prevent multiple recordings from being silently
    # pooled without visibility. Resampling is disabled here to avoid duplicating runtime;
    # the inferential/bootstrap results remain in subject_window_summary.csv.
    session_rows = []
    session_cfg = replace(cfg, bootstrap_iterations=0, permutation_iterations=0)
    for seed_offset, ((subject, session, window), d) in enumerate(
        trial_df.groupby(["subject", "session", "window_sec"], observed=True)
    ):
        try:
            session_summary, _, _ = analyze_subject_window(d, session_cfg, 900000 + seed_offset)
            session_summary["session"] = str(session)
            session_rows.append(session_summary)
        except ValueError:
            continue
    session_summary_df = pd.DataFrame(session_rows)
    if not session_summary_df.empty:
        first_cols = ["subject", "session", "window_sec"]
        other_cols = [c for c in session_summary_df.columns if c not in first_cols]
        session_summary_df = session_summary_df[first_cols + other_cols].sort_values(first_cols)

    summary_df.to_csv(table_dir / "subject_window_summary.csv", index=False, encoding="utf-8-sig")
    if not session_summary_df.empty:
        session_summary_df.to_csv(table_dir / "session_window_summary.csv", index=False, encoding="utf-8-sig")
    phase_df_all.to_csv(table_dir / "phase_level_summary.csv", index=False, encoding="utf-8-sig")
    trend_df.to_csv(table_dir / "window_effect_summary.csv", index=False, encoding="utf-8-sig")

    # Per-subject figures.
    selected_window = select_classification_window(summary_df, cfg.classification_window_sec)
    for subject, subject_summary in summary_df.groupby("subject", observed=True):
        subdir = cfg.output_dir / "figures" / str(subject)
        subdir.mkdir(parents=True, exist_ok=True)
        trend_row = trend_df[trend_df["subject"] == subject].iloc[0]
        plot_window_effect(subject_summary, trend_row, subdir / "window_effect.png", cfg.plot_dpi)

        # Detailed plots at every window, with confusion/capacity emphasized at selected window.
        for _, row in subject_summary.iterrows():
            window = float(row["window_sec"])
            details = details_by_key[(str(subject), window)]
            p = phase_df_all[(phase_df_all["subject"] == subject) & np.isclose(phase_df_all["window_sec"], window)]
            tag = f"T{window:.2f}s".replace(".", "p")
            plot_transfer(details, row.to_dict(), subdir / f"transfer_{tag}.png", cfg.plot_dpi)
            plot_theta_by_phase(p, row.to_dict(), subdir / f"theta_by_phase_{tag}.png", cfg.plot_dpi)
            plot_residuals(details, row.to_dict(), subdir / f"residuals_{tag}.png", cfg.plot_dpi)
            if np.isclose(window, selected_window):
                plot_confusion(details, p, row.to_dict(), subdir / f"confusion_{tag}.png", cfg.plot_dpi)
                plot_capacity(row.to_dict(), subdir / f"capacity_{tag}.png", cfg.plot_dpi, cfg)
                plot_polar(details, p, row.to_dict(), subdir / f"polar_{tag}.png", cfg.plot_dpi)

    # Group figures are also meaningful for one subject and keep output structure constant.
    plot_group_window_effect(summary_df, cfg.output_dir / "figures" / "group_sigma_vs_window.png", cfg.plot_dpi)
    plot_group_mmax(summary_df, cfg.classification_window_sec, cfg.output_dir / "figures" / "group_mmax.png", cfg.plot_dpi)

    write_markdown_report(
        cfg.output_dir / "Stage1_analysis_report.md",
        cfg,
        file_qc_df,
        summary_df,
        trend_df,
    )

    print("[Stage1 analysis] Finished.")
    print(f"  Output: {cfg.output_dir.resolve()}")
    print(f"  Report: {(cfg.output_dir / 'Stage1_analysis_report.md').resolve()}")


# =============================================================================
# 9. Command-line interface
# =============================================================================


def parse_int_list(values: list[int]) -> tuple[int, ...]:
    return tuple(values)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze Stage-1 SSVEP phase-transfer MAT files.")
    p.add_argument("--root", type=Path, default=Path("."), help="Root containing sub* folders or MAT files")
    p.add_argument("--output", type=Path, default=Path("stage1_analysis"))
    p.add_argument("--subject-glob", default="sub*")
    p.add_argument("--file-glob", default="Stage1_SSVEP_PhaseTransfer_*.mat")
    p.add_argument("--no-recursive", action="store_true")

    p.add_argument("--fs-override", type=float, default=None)
    p.add_argument("--stim-freq-override", type=float, default=None)
    p.add_argument("--phase-count-override", type=int, default=None)
    p.add_argument("--phase-values-deg", type=float, nargs="+", default=None)
    p.add_argument("--trigger-base-override", type=int, default=None)
    p.add_argument("--block-count", type=int, default=None)
    p.add_argument("--trials-per-phase-per-block", type=int, default=None)

    p.add_argument("--marker-channel", type=int, default=DEFAULT_MARKER_CHANNEL, help="Zero-based marker column")
    p.add_argument("--eeg-channels", type=int, nargs="+", default=list(DEFAULT_EEG_CHANNELS), help="Zero-based EEG columns")
    p.add_argument("--channel-names", nargs="+", default=None)
    p.add_argument("--reference-channel", type=int, default=DEFAULT_REFERENCE_CHANNEL, help="Absolute zero-based data column")
    p.add_argument("--spatial-mode", choices=["single", "mean", "phase_aligned"], default="phase_aligned")
    p.add_argument("--rereference", choices=["none", "average"], default="none")
    p.add_argument(
        "--linked-mastoids-reference",
        action="store_true",
        help="Subtract mean(TP9, TP10) from selected EEG channels before filtering and phase analysis",
    )

    p.add_argument("--transient", type=float, default=0.30)
    p.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--bandpass-low", type=float, default=5.0)
    p.add_argument("--bandpass-high", type=float, default=45.0)
    p.add_argument("--notch", type=float, default=50.0, help="Use 0 to disable")
    p.add_argument("--filter-order", type=int, default=4)

    p.add_argument("--event-tolerance", type=float, default=0.080)
    p.add_argument("--allow-timing-fallback", action="store_true")
    p.add_argument("--manual-first-onset-sec", type=float, default=None)
    p.add_argument("--min-event-match-fraction", type=float, default=0.80)

    p.add_argument("--artifact-mad-threshold", type=float, default=8.0)
    p.add_argument("--absolute-p2p-limit", type=float, default=None)
    p.add_argument("--reject-artifacts", action="store_true")

    p.add_argument("--classification-window", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260714)
    p.add_argument("--capacity-threshold", type=float, default=0.10)
    p.add_argument("--capacity-max-m", type=int, default=30)
    return p


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        root_dir=args.root,
        output_dir=args.output,
        subject_glob=args.subject_glob,
        file_glob=args.file_glob,
        recursive=not args.no_recursive,
        fs_override=args.fs_override,
        stim_freq_override=args.stim_freq_override,
        phase_count_override=args.phase_count_override,
        phase_values_deg_override=None if args.phase_values_deg is None else tuple(args.phase_values_deg),
        trigger_base_override=args.trigger_base_override,
        block_count=args.block_count,
        trials_per_phase_per_block=args.trials_per_phase_per_block,
        marker_channel=args.marker_channel,
        eeg_channels=parse_int_list(args.eeg_channels),
        channel_names=DEFAULT_CHANNEL_NAMES if args.channel_names is None else tuple(args.channel_names),
        reference_channel=args.reference_channel,
        spatial_mode=args.spatial_mode,
        rereference=args.rereference,
        linked_mastoids_reference=args.linked_mastoids_reference,
        transient_sec=args.transient,
        window_lengths_sec=tuple(args.windows),
        bandpass_low_hz=args.bandpass_low,
        bandpass_high_hz=args.bandpass_high,
        notch_hz=None if args.notch == 0 else args.notch,
        filter_order=args.filter_order,
        event_match_tolerance_sec=args.event_tolerance,
        allow_timing_fallback=args.allow_timing_fallback,
        manual_first_onset_sec=args.manual_first_onset_sec,
        min_event_match_fraction=args.min_event_match_fraction,
        artifact_mad_threshold=args.artifact_mad_threshold,
        absolute_peak_to_peak_limit=args.absolute_p2p_limit,
        reject_artifacts=args.reject_artifacts,
        classification_window_sec=args.classification_window,
        bootstrap_iterations=args.bootstrap,
        permutation_iterations=args.permutations,
        random_seed=args.seed,
        capacity_error_threshold=args.capacity_threshold,
        capacity_max_m=args.capacity_max_m,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    try:
        run_analysis(cfg)
    except Exception as exc:
        print(f"[Stage1 analysis] ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
