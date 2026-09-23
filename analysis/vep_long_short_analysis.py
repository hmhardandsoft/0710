#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse the long/short transient VEP recordings saved by vep1.0_long_short.py.

The default input layout is VEP_Dataset/subN/*.mat. Each MAT file is one block.
Only TransientVEP_1min files with an explicit long/short token are selected.

Examples
--------
python analysis/vep_long_short_analysis.py
python analysis/vep_long_short_analysis.py --input-root . --reference TP9,TP10
python analysis/vep_long_short_analysis.py --reference Oz --roi POz,O1,O2

MAT files contain no channel labels or physical voltage unit. Check the
configured channel mapping against the acquisition montage before interpreting
amplitudes. Latencies are relative to the electrical trigger, not a measured
photodiode onset.

Outputs include file/block QC, subject feature and recovery CSV tables, NPZ
waveforms, and long/short ERP plots. Plots are PNG with matplotlib or standalone
SVG when matplotlib is unavailable. The short baseline-corrected ERP is a
descriptive average because its prestimulus interval contains prior responses;
recovery features use the continuous-data deconvolution model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import BSpline
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt
from scipy.sparse import coo_matrix, diags, hstack, vstack
from scipy.sparse.linalg import lsmr


DEFAULT_CHANNELS = ("FCz", "TP9", "Pz", "POz", "O1", "Oz", "O2", "TP10")
COMPONENTS = {"P1": (0.060, 0.130), "N1": (0.130, 0.210), "P2": (0.210, 0.350)}
SHORT_PREDICTION_SOA_MS = (70, 80, 90, 100, 110, 120, 130)


@dataclass(frozen=True)
class Config:
    input_root: Path = Path("VEP_Dataset")
    output_root: Path = Path("analysis_results")
    channel_names: tuple[str, ...] = DEFAULT_CHANNELS
    eeg_columns: tuple[int, ...] = tuple(range(8))
    marker_column: int = 8
    reference: tuple[str, ...] = ("TP9", "TP10")
    roi: tuple[str, ...] = ("POz", "O1", "Oz", "O2")
    oz_channel: str = "Oz"
    low_hz: float = 0.1
    high_hz: float = 40.0
    notch_hz: float = 50.0
    target_fs: float = 250.0
    marker_tolerance: float = 0.2
    min_event_gap_ms: float = 20.0
    artifact_mad_multiplier: float = 8.0
    long_pre: float = -0.200
    long_post: float = 0.800
    long_baseline: tuple[float, float] = (-0.200, -0.020)
    short_pre: float = -0.050
    short_post: float = 0.400
    short_baseline: tuple[float, float] = (-0.050, -0.010)
    deconv_post: float = 0.400
    ridge_alpha: float = 3.0


@dataclass
class Block:
    subject: str
    mode: str
    path: Path
    original_fs: float
    fs: float
    signals: np.ndarray  # samples x [ROI, Oz], referenced and filtered
    events: np.ndarray  # event sample indices at fs
    valid: np.ndarray  # continuous samples accepted for regression/epochs
    raw_event_count: int
    median_soa_ms: float
    outside_expected_soa: int
    bad_fraction: float
    repaired_impulsive_samples: int
    epoch_times: np.ndarray | None = None
    epoch_mean: np.ndarray | None = None
    corrected_mean: np.ndarray | None = None
    kept_epoch_count: int = 0


def parse_names(value: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in value.split(",") if piece.strip())


def parse_columns(value: str) -> tuple[int, ...]:
    try:
        columns = tuple(int(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("EEG columns must be comma-separated integers") from exc
    if len(columns) != 8 or len(set(columns)) != 8 or min(columns) < 0:
        raise argparse.ArgumentTypeError("Specify exactly eight distinct nonnegative EEG columns")
    return columns


def parse_interval(value: str) -> tuple[float, float]:
    try:
        a, b = (float(v.strip()) / 1000.0 for v in value.split(","))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Specify two comma-separated times in ms") from exc
    if not np.isfinite([a, b]).all() or a >= b:
        raise argparse.ArgumentTypeError("Baseline start must be before its end")
    return a, b


def validate_config(cfg: Config) -> None:
    if len(cfg.channel_names) != 8 or len(set(cfg.channel_names)) != 8:
        raise ValueError("--channel-names must list eight distinct EEG names")
    if len(cfg.eeg_columns) != 8 or len(set(cfg.eeg_columns)) != 8:
        raise ValueError("--eeg-columns must list eight distinct EEG columns")
    if cfg.marker_column in cfg.eeg_columns or cfg.marker_column < 0:
        raise ValueError("The marker column must be distinct from EEG columns")
    if not cfg.roi or any(name not in cfg.channel_names for name in cfg.roi):
        raise ValueError("All ROI names must be in --channel-names")
    if cfg.oz_channel not in cfg.channel_names:
        raise ValueError("The Oz display channel must be in --channel-names")
    if any(name not in cfg.channel_names for name in cfg.reference):
        raise ValueError("All reference names must be in --channel-names")
    if len(set(cfg.reference)) != len(cfg.reference):
        raise ValueError("Reference names must not repeat")
    if cfg.low_hz <= 0 or cfg.high_hz <= cfg.low_hz or cfg.target_fs <= 2 * cfg.high_hz:
        raise ValueError("Invalid filter band or target sampling rate")
    for label, interval, pre, post in (
        ("long", cfg.long_baseline, cfg.long_pre, cfg.long_post),
        ("short", cfg.short_baseline, cfg.short_pre, cfg.short_post),
    ):
        if interval[0] < pre or interval[1] > min(post, 0):
            raise ValueError(f"{label} baseline must lie within the prestimulus epoch")
    if cfg.ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive")
    if set(cfg.roi).issubset(cfg.reference) or cfg.oz_channel in cfg.reference:
        warnings.warn("A displayed channel is part of the reference; inspect its waveform carefully")


def discover_files(root: Path) -> dict[str, dict[str, list[Path]]]:
    found: dict[str, dict[str, list[Path]]] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir() or not re.fullmatch(r"sub\d+", subject_dir.name, re.I):
            continue
        modes = {"long": [], "short": []}
        for path in sorted(subject_dir.glob("TransientVEP_1min_*.mat")):
            tokens = path.stem.lower().split("_")
            matched = [mode for mode in modes if mode in tokens]
            if len(matched) == 1:
                modes[matched[0]].append(path)
        if any(modes.values()):
            found[subject_dir.name] = modes
    if not found:
        raise FileNotFoundError(f"No long/short TransientVEP MAT files found in {root}/subN")
    return found


def detect_events(marker: np.ndarray, fs: float, cfg: Config) -> np.ndarray:
    finite = np.isfinite(marker)
    pulse = finite & (np.abs(marker - 1.0) <= cfg.marker_tolerance)
    candidates = np.flatnonzero(pulse & ~np.r_[False, pulse[:-1]])
    min_gap = max(1, round(fs * cfg.min_event_gap_ms / 1000.0))
    kept: list[int] = []
    for sample in candidates:
        if not kept or sample - kept[-1] >= min_gap:
            kept.append(int(sample))
    return np.asarray(kept, dtype=int)


def interpolate_nonfinite(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    invalid = ~np.isfinite(x).all(axis=1)
    if not invalid.any():
        return x, invalid
    x = x.copy()
    sample_index = np.arange(len(x))
    for channel in range(x.shape[1]):
        finite = np.isfinite(x[:, channel])
        if finite.sum() < 2:
            raise ValueError("EEG channel contains fewer than two finite samples")
        x[:, channel] = np.interp(sample_index, sample_index[finite], x[finite, channel])
    return x, invalid


def repair_impulsive_samples(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate isolated amplifier spikes before zero-phase filtering rings."""
    local_median = median_filter(x, size=(11, 1), mode="mirror")
    residual = x - local_median
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    threshold = 15.0 * np.maximum(scale, 1e-9)
    spikes = np.abs(residual) > threshold[None, :]
    if not spikes.any():
        return x, np.zeros(len(x), dtype=bool)
    x = x.copy()
    sample_index = np.arange(len(x))
    for channel in range(x.shape[1]):
        if not spikes[:, channel].any():
            continue
        good = ~spikes[:, channel]
        if good.sum() < 2:
            raise ValueError("EEG channel has too many impulsive outliers")
        x[~good, channel] = np.interp(sample_index[~good], sample_index[good], x[good, channel])
    return x, spikes.any(axis=1)


def robust_artifact_mask(signals: np.ndarray, fs: float, multiplier: float) -> np.ndarray:
    """Flag 100 ms windows with unusually large ROI/Oz peak-to-peak range."""
    n = len(signals)
    width = max(2, round(fs * 0.1))
    scores = []
    slices = []
    for start in range(0, n, width):
        stop = min(start + width, n)
        segment = signals[start:stop]
        if len(segment) < 2:
            continue
        scores.append(float(np.max(np.ptp(segment, axis=0))))
        slices.append((start, stop))
    if not scores:
        return np.ones(n, dtype=bool)
    scores_array = np.asarray(scores)
    median = float(np.median(scores_array))
    mad = float(1.4826 * np.median(np.abs(scores_array - median)))
    threshold = median + multiplier * max(mad, median * 0.05, 1e-12)
    valid = np.ones(n, dtype=bool)
    pad = round(fs * 0.05)
    for (start, stop), score in zip(slices, scores_array):
        if score > threshold:
            valid[max(0, start - pad) : min(n, stop + pad)] = False
    return valid


def load_block(subject: str, mode: str, path: Path, cfg: Config) -> Block:
    mat = loadmat(path)
    if "data" not in mat or "Fs" not in mat:
        raise ValueError("MAT file must contain data and Fs")
    raw = np.asarray(mat["data"], dtype=float)
    fs_values = np.asarray(mat["Fs"], dtype=float).ravel()
    if fs_values.size != 1 or not np.isfinite(fs_values[0]) or fs_values[0] <= 0:
        raise ValueError("Fs must be one positive scalar")
    fs_original = float(fs_values[0])
    required_column = max(*cfg.eeg_columns, cfg.marker_column)
    if raw.ndim != 2 or raw.shape[0] <= raw.shape[1] or raw.shape[1] <= required_column:
        raise ValueError(f"Expected samples x channels including column {required_column}; got {raw.shape}")
    raw_events = detect_events(raw[:, cfg.marker_column], fs_original, cfg)
    if len(raw_events) < 3:
        raise ValueError(f"Only {len(raw_events)} trigger events detected")
    soas = np.diff(raw_events) / fs_original * 1000.0
    expected_range = (1100.0, 2100.0) if mode == "long" else (60.0, 135.0)
    outside = int(np.sum((soas < expected_range[0] - 5) | (soas > expected_range[1] + 5)))

    eeg, invalid_raw = interpolate_nonfinite(raw[:, cfg.eeg_columns])
    name_to_position = {name: i for i, name in enumerate(cfg.channel_names)}
    if cfg.reference:
        reference_positions = [name_to_position[name] for name in cfg.reference]
        eeg = eeg - eeg[:, reference_positions].mean(axis=1, keepdims=True)
    eeg, impulsive_raw = repair_impulsive_samples(eeg)
    if cfg.notch_hz > 0 and cfg.notch_hz < fs_original / 2:
        b, a = iirnotch(cfg.notch_hz, 30, fs=fs_original)
        eeg = filtfilt(b, a, eeg, axis=0)
    if cfg.high_hz >= fs_original / 2:
        raise ValueError(f"High cutoff {cfg.high_hz} Hz is above Nyquist for Fs={fs_original}")
    sos = butter(4, [cfg.low_hz, cfg.high_hz], btype="bandpass", fs=fs_original, output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=0)
    fraction = Fraction(cfg.target_fs / fs_original).limit_denominator(1000)
    up, down = fraction.numerator, fraction.denominator
    effective_fs = fs_original * up / down
    if abs(effective_fs - cfg.target_fs) > 0.01:
        raise ValueError("Cannot resample to target_fs accurately; choose a compatible rate")
    eeg = resample_poly(eeg, up, down, axis=0)
    roi_positions = [name_to_position[name] for name in cfg.roi]
    oz_position = name_to_position[cfg.oz_channel]
    signals = np.column_stack((eeg[:, roi_positions].mean(axis=1), eeg[:, oz_position]))
    events = np.rint(raw_events * up / down).astype(int)
    if len(np.unique(events)) != len(events):
        raise ValueError("Event samples collided after resampling")
    valid = robust_artifact_mask(signals, effective_fs, cfg.artifact_mad_multiplier)
    if invalid_raw.any() or impulsive_raw.any():
        bad_original = np.flatnonzero(invalid_raw | impulsive_raw)
        bad_resampled = np.rint(bad_original * up / down).astype(int)
        pad = round(effective_fs * 0.05)
        for sample in bad_resampled:
            valid[max(0, sample - pad) : min(len(valid), sample + pad + 1)] = False
    return Block(
        subject=subject,
        mode=mode,
        path=path,
        original_fs=fs_original,
        fs=effective_fs,
        signals=signals,
        events=events,
        valid=valid,
        raw_event_count=len(raw_events),
        median_soa_ms=float(np.median(soas)),
        outside_expected_soa=outside,
        bad_fraction=float(1.0 - valid.mean()),
        repaired_impulsive_samples=int(impulsive_raw.sum()),
    )


def epoch_average(block: Block, cfg: Config) -> None:
    pre, post, baseline = (
        (cfg.long_pre, cfg.long_post, cfg.long_baseline)
        if block.mode == "long"
        else (cfg.short_pre, cfg.short_post, cfg.short_baseline)
    )
    start, stop = round(pre * block.fs), round(post * block.fs)
    offsets = np.arange(start, stop + 1)
    times = offsets / block.fs
    baseline_mask = (times >= baseline[0] - 1e-9) & (times <= baseline[1] + 1e-9)
    if baseline_mask.sum() < 2:
        raise ValueError("Baseline window contains fewer than two samples")
    uncorrected = []
    corrected = []
    for event in block.events:
        indices = event + offsets
        if indices[0] < 0 or indices[-1] >= len(block.signals):
            continue
        if not block.valid[indices].all():
            continue
        epoch = block.signals[indices]
        uncorrected.append(epoch)
        corrected.append(epoch - epoch[baseline_mask].mean(axis=0, keepdims=True))
    block.epoch_times = times
    block.kept_epoch_count = len(uncorrected)
    if uncorrected:
        block.epoch_mean = np.mean(uncorrected, axis=0)
        block.corrected_mean = np.mean(corrected, axis=0)


def equal_block_average(blocks: list[Block], field: str) -> tuple[np.ndarray, np.ndarray, list[Block]] | None:
    usable = [block for block in blocks if getattr(block, field) is not None]
    if not usable:
        return None
    first_times = usable[0].epoch_times
    compatible = [block for block in usable if np.array_equal(block.epoch_times, first_times)]
    if len(compatible) != len(usable):
        warnings.warn("Some blocks have incompatible sampling grids and were excluded from the average")
    return first_times, np.mean([getattr(block, field) for block in compatible], axis=0), compatible


def component_metrics(times: np.ndarray, signal: np.ndarray, label: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, (a, b) in COMPONENTS.items():
        mask = (times >= a) & (times <= b)
        if mask.sum() < 2:
            result[f"{label}_{name}_mean"] = float("nan")
            result[f"{label}_{name}_latency_ms"] = float("nan")
            continue
        segment = signal[mask]
        indices = np.flatnonzero(mask)
        peak_local = int(np.argmin(segment) if name == "N1" else np.argmax(segment))
        result[f"{label}_{name}_mean"] = float(np.mean(segment))
        result[f"{label}_{name}_latency_ms"] = float(times[indices[peak_local]] * 1000)
    return result


def spline_history(soa_ms: np.ndarray) -> np.ndarray:
    """Centered cubic B-spline history terms; 100 ms is the reference level."""
    knots = np.array([50.0] * 4 + [150.0] * 4)
    basis = BSpline.design_matrix(np.clip(soa_ms, 50, 150), knots, 3).toarray()
    center = BSpline.design_matrix(np.array([100.0]), knots, 3).toarray()[0]
    return (basis - center)[:, :3]


def deconv_design(blocks: list[Block], cfg: Config) -> tuple[coo_matrix, np.ndarray, np.ndarray, int]:
    """Continuous FIR design: event, two SOA histories, event order, block drift."""
    fs = blocks[0].fs
    if any(abs(block.fs - fs) > 1e-6 for block in blocks):
        raise ValueError("Short blocks must share the same resampled Fs")
    lags = np.arange(round(cfg.deconv_post * fs) + 1, dtype=int)
    n_lags = len(lags)
    n_event_features = 8  # intercept + 3 previous + 3 previous-previous + order
    n_kernel = n_lags * n_event_features
    total_samples = sum(len(block.signals) for block in blocks)
    n_columns = n_kernel + 2 * len(blocks)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    y = np.concatenate([block.signals for block in blocks], axis=0)
    valid = np.concatenate([block.valid for block in blocks])
    offset = 0
    for block_index, block in enumerate(blocks):
        n_samples = len(block.signals)
        event_indices = block.events
        n_events = len(event_indices)
        history = np.zeros((n_events, n_event_features), dtype=float)
        history[:, 0] = 1.0
        intervals = np.diff(event_indices) / fs * 1000.0
        if n_events > 1:
            history[1:, 1:4] = spline_history(intervals)
        if n_events > 2:
            history[2:, 4:7] = spline_history(intervals[:-1])
        history[:, 7] = np.linspace(-0.5, 0.5, n_events)
        local_rows = event_indices[:, None] + lags[None, :]
        in_range = local_rows < n_samples
        selected_rows = (local_rows + offset)[in_range]
        lag_indices = np.broadcast_to(lags, local_rows.shape)[in_range]
        event_grid = np.broadcast_to(np.arange(n_events)[:, None], local_rows.shape)[in_range]
        for feature in range(n_event_features):
            feature_values = history[event_grid, feature]
            nonzero = feature_values != 0
            rows.append(selected_rows[nonzero])
            columns.append(feature * n_lags + lag_indices[nonzero])
            values.append(feature_values[nonzero])
        sample_rows = np.arange(n_samples, dtype=int) + offset
        rows.extend((sample_rows, sample_rows))
        columns.extend((np.full(n_samples, n_kernel + 2 * block_index),
                        np.full(n_samples, n_kernel + 2 * block_index + 1)))
        values.extend((np.ones(n_samples), np.linspace(-0.5, 0.5, n_samples)))
        offset += n_samples
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(total_samples, n_columns),
    ).tocsr()
    # Keep rows only when every contributing block marks that sample as valid.
    return matrix[valid], y[valid], lags / fs, n_kernel


def fit_short_deconvolution(blocks: list[Block], cfg: Config) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    matrix, y, times, n_kernel = deconv_design(blocks, cfg)
    if matrix.shape[0] < 10 * n_kernel:
        raise ValueError("Too few clean continuous samples for short deconvolution")
    # The dense short event train makes the FIR baseline weakly identifiable.
    # Anchor its first 32 ms and last 40 ms to zero in addition to ridge
    # regularization; block intercepts/drifts remain unpenalized.
    n_lags = len(times)
    penalty_weights = np.full(n_kernel, cfg.ridge_alpha, dtype=float)
    anchor = (times <= 0.032) | (times >= cfg.deconv_post - 0.040)
    for feature in range(8):
        penalty_weights[feature * n_lags : (feature + 1) * n_lags][anchor] *= 100.0
    penalty = hstack((diags(np.sqrt(penalty_weights), format="csr"),
                      coo_matrix((n_kernel, matrix.shape[1] - n_kernel))), format="csr")
    augmented = vstack((matrix, penalty), format="csr")
    rhs_zeros = np.zeros(penalty.shape[0])
    coefficients = np.column_stack([
        lsmr(augmented, np.r_[y[:, ch], rhs_zeros], atol=1e-5, btol=1e-5, maxiter=400)[0]
        for ch in range(y.shape[1])
    ])
    kernels = coefficients[:n_kernel].reshape(8, n_lags, y.shape[1])
    predicted: dict[int, np.ndarray] = {}
    for soa in SHORT_PREDICTION_SOA_MS:
        basis = spline_history(np.array([soa]))[0]
        waveform = kernels[0].copy()
        for term in range(3):
            waveform += basis[term] * kernels[1 + term]
        predicted[soa] = waveform
    residual = y - matrix @ coefficients
    explained_variance = 1.0 - float(np.sum(residual[:, 0] ** 2) / np.sum((y[:, 0] - y[:, 0].mean()) ** 2))
    return times, predicted, explained_variance


def fit_recovery_tau(soas_ms: np.ndarray, amplitudes: np.ndarray) -> float:
    """Return NaN unless a simple saturating recovery curve is well supported."""
    if len(soas_ms) < 5 or not np.isfinite(amplitudes).all():
        return float("nan")
    direction = np.sign(np.median(amplitudes))
    positive = amplitudes * direction
    if direction == 0 or np.min(positive) <= 0 or np.corrcoef(soas_ms, positive)[0, 1] < 0.75:
        return float("nan")
    def model(x: np.ndarray, asymptote: float, tau: float) -> np.ndarray:
        return asymptote * (1.0 - np.exp(-x / tau))
    try:
        params, covariance = curve_fit(model, soas_ms, positive, p0=[max(positive) * 2, 100.0],
                                       bounds=([0, 10], [max(positive) * 100, 1000]), maxfev=10000)
    except (RuntimeError, ValueError):
        return float("nan")
    residual = positive - model(soas_ms, *params)
    variance = np.sum((positive - positive.mean()) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / variance if variance > 0 else float("nan")
    tau_error = float(np.sqrt(covariance[1, 1])) if np.isfinite(covariance[1, 1]) else float("inf")
    return float(params[1]) if r2 >= 0.8 and 10 < params[1] < 1000 and tau_error / params[1] <= 0.5 else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_svg_panels(path: Path, title: str, xlabel: str,
                    panels: list[tuple[str, list[tuple[np.ndarray, np.ndarray, str, float, str]]]],
                    xlim: tuple[float, float] | None = None,
                    baseline: tuple[float, float] | None = None,
                    markers: bool = False) -> None:
    """Dependency-free vector plot fallback when matplotlib is unavailable."""
    width, height = 1000, (700 if len(panels) == 2 else 430)
    left, right, top, panel_height, gap = 105, 40, 78, 240, 75
    plot_width = width - left - right
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" '
             f'font-size="20">{escape(title)}</text>']
    for panel_index, (ylabel, series) in enumerate(panels):
        plot_top = top + panel_index * (panel_height + gap)
        all_x = np.concatenate([np.asarray(line[0]) for line in series])
        xmin, xmax = xlim if xlim else (float(np.min(all_x)), float(np.max(all_x)))
        visible_y = [np.asarray(y)[(np.asarray(x) >= xmin) & (np.asarray(x) <= xmax)] for x, y, *_ in series]
        all_y = np.concatenate(visible_y)
        finite_y = all_y[np.isfinite(all_y)]
        if finite_y.size == 0:
            continue
        ymin, ymax = float(np.min(finite_y)), float(np.max(finite_y))
        padding = max((ymax - ymin) * 0.12, 1e-6)
        ymin, ymax = ymin - padding, ymax + padding
        xcoord = lambda x: left + (float(x) - xmin) / (xmax - xmin) * plot_width
        ycoord = lambda y: plot_top + panel_height - (float(y) - ymin) / (ymax - ymin) * panel_height
        if baseline:
            b0, b1 = baseline
            x0, x1 = max(left, xcoord(b0)), min(left + plot_width, xcoord(b1))
            if x1 > x0:
                items.append(f'<rect x="{x0:.2f}" y="{plot_top}" width="{x1-x0:.2f}" '
                             f'height="{panel_height}" fill="#e6f0f9"/>')
        for tick in np.linspace(ymin, ymax, 5):
            yy = ycoord(tick)
            items.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_width}" y2="{yy:.2f}" '
                         f'stroke="#e7e7e7"/>')
            items.append(f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" '
                         f'font-family="Arial" font-size="12">{tick:.1f}</text>')
        if xmin <= 0 <= xmax:
            xx = xcoord(0)
            items.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" '
                         f'y2="{plot_top+panel_height}" stroke="#666" stroke-dasharray="5,4"/>')
        items.append(f'<rect x="{left}" y="{plot_top}" width="{plot_width}" height="{panel_height}" '
                     f'fill="none" stroke="#333"/>')
        for tick in np.linspace(xmin, xmax, 6):
            xx = xcoord(tick)
            items.append(f'<line x1="{xx:.2f}" y1="{plot_top+panel_height}" x2="{xx:.2f}" '
                         f'y2="{plot_top+panel_height+5}" stroke="#333"/>')
            items.append(f'<text x="{xx:.2f}" y="{plot_top+panel_height+20}" text-anchor="middle" '
                         f'font-family="Arial" font-size="12">{tick:.0f}</text>')
        items.append(f'<text transform="translate(24 {plot_top+panel_height/2:.2f}) rotate(-90)" '
                     f'text-anchor="middle" font-family="Arial" font-size="14">{escape(ylabel)}</text>')
        legend_index = 0
        for x_values, y_values, color, stroke_width, label in series:
            x_array = np.asarray(x_values)
            y_array = np.asarray(y_values)
            inside = np.isfinite(x_array) & np.isfinite(y_array) & (x_array >= xmin) & (x_array <= xmax)
            points = " ".join(f"{xcoord(x):.2f},{ycoord(y):.2f}" for x, y in zip(x_array[inside], y_array[inside]))
            if points:
                items.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
                             f'points="{points}"/>')
            if markers:
                for x, y in zip(x_array[inside], y_array[inside]):
                    items.append(f'<circle cx="{xcoord(x):.2f}" cy="{ycoord(y):.2f}" r="3" fill="{color}"/>')
            if label:
                # Plot labels are drawn in panel order near the upper right.
                ly = plot_top + 18 + 18 * legend_index
                lx = left + plot_width - 175
                items.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+24}" y2="{ly-4}" '
                             f'stroke="{color}" stroke-width="{stroke_width}"/>')
                items.append(f'<text x="{lx+30}" y="{ly}" font-family="Arial" font-size="12">'
                             f'{escape(label)}</text>')
                legend_index += 1
    items.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" '
                 f'font-family="Arial" font-size="14">{escape(xlabel)}</text>')
    items.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items), encoding="utf-8")


def plot_averages(path: Path, title: str, times: np.ndarray, blocks: list[Block],
                  average: np.ndarray, field: str, baseline: tuple[float, float] | None) -> None:
    if path.suffix == ".svg":
        panels = []
        for channel, label in ((0, "ROI"), (1, "Oz")):
            series = [(times * 1000, getattr(block, field)[:, channel], "#aaaaaa", 1.0, "")
                      for block in blocks]
            series.append((times * 1000, average[:, channel], "#155f9e", 2.5, "Subject mean"))
            panels.append((f"{label} (device units)", series))
        plot_svg_panels(path, title, "Time from trigger (ms)", panels,
                        baseline=None if baseline is None else (baseline[0] * 1000, baseline[1] * 1000))
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for channel, ax in enumerate(axes):
        for block in blocks:
            ax.plot(times * 1000, getattr(block, field)[:, channel], color="0.68", lw=1.0, alpha=0.8)
        ax.plot(times * 1000, average[:, channel], color="#155f9e", lw=2.0, label="Subject mean")
        ax.axvline(0, color="0.25", ls="--", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        if baseline is not None:
            ax.axvspan(baseline[0] * 1000, baseline[1] * 1000, color="#dbe9f5", alpha=0.6)
        ax.set_ylabel(("ROI" if channel == 0 else "Oz") + " (device units)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time from trigger (ms)")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_deconvolution(path: Path, long_times: np.ndarray, long_mean: np.ndarray,
                       short_times: np.ndarray, predicted: dict[int, np.ndarray]) -> None:
    if path.suffix == ".svg":
        panels = []
        for channel, label in ((0, "ROI"), (1, "Oz")):
            panels.append((f"{label} (device units)", [
                (long_times * 1000, long_mean[:, channel], "#155f9e", 2.5, "Long average"),
                (short_times * 1000, predicted[100][:, channel], "#bd412c", 2.5, "Short, SOA 100 ms"),
            ]))
        plot_svg_panels(path, "Long VEP and short deconvolved response", "Time from trigger (ms)",
                        panels, xlim=(0, 400))
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for channel, ax in enumerate(axes):
        ax.plot(long_times * 1000, long_mean[:, channel], color="#155f9e", lw=2,
                label="Long average")
        ax.plot(short_times * 1000, predicted[100][:, channel], color="#bd412c", lw=2,
                label="Short deconvolved, previous SOA 100 ms")
        ax.axvline(0, color="0.25", ls="--", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlim(0, 400)
        ax.set_ylabel(("ROI" if channel == 0 else "Oz") + " (device units)")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time from trigger (ms)")
    fig.suptitle("Long VEP and short deconvolved response")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_recovery(path: Path, rows: list[dict[str, object]]) -> None:
    if path.suffix == ".svg":
        colors = {"P1": "#155f9e", "N1": "#bd412c", "P2": "#45844d"}
        series = []
        for component in COMPONENTS:
            chosen = [row for row in rows if row["component"] == component and row["channel"] == "ROI"]
            if chosen:
                series.append((np.array([row["soa_ms"] for row in chosen]),
                               np.array([row["short_mean"] for row in chosen]),
                               colors[component], 2.5, component))
        plot_svg_panels(path, "Short VEP recovery curve, ROI", "Previous onset interval (ms)",
                        [("Deconvolved component mean (device units)", series)], markers=True)
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for component in COMPONENTS:
        chosen = [row for row in rows if row["component"] == component and row["channel"] == "ROI"]
        if chosen:
            ax.plot([row["soa_ms"] for row in chosen], [row["short_mean"] for row in chosen],
                    marker="o", label=component)
    ax.set_xlabel("Previous onset-to-onset interval (ms)")
    ax.set_ylabel("Deconvolved component mean (device units)")
    ax.set_title("Short VEP recovery curve, ROI")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyse_subject(subject: str, modes: dict[str, list[Path]], cfg: Config,
                    file_qc: list[dict[str, object]], feature_rows: list[dict[str, object]],
                    recovery_rows: list[dict[str, object]], plot_extension: str) -> None:
    blocks: dict[str, list[Block]] = {"long": [], "short": []}
    for mode, files in modes.items():
        for path in files:
            try:
                block = load_block(subject, mode, path, cfg)
                epoch_average(block, cfg)
                blocks[mode].append(block)
                file_qc.append({"subject": subject, "mode": mode, "file": str(path), "status": "ok",
                                "fs_original": block.original_fs,
                                "fs_analysis": block.fs, "events": block.raw_event_count,
                                "epochs_kept": block.kept_epoch_count,
                                "valid_epoch_fraction": block.kept_epoch_count / block.raw_event_count,
                                "bad_sample_fraction": block.bad_fraction,
                                "repaired_impulsive_samples": block.repaired_impulsive_samples,
                                "median_soa_ms": block.median_soa_ms,
                                "outside_expected_soa": block.outside_expected_soa})
                print(f"  {mode:5s} {path.name}: {block.kept_epoch_count}/{block.raw_event_count} epochs", flush=True)
            except (ValueError, KeyError, OSError) as exc:
                file_qc.append({"subject": subject, "mode": mode, "file": str(path), "status": "failed",
                                "reason": str(exc)})
                print(f"  WARNING: skipping {path.name}: {exc}", file=sys.stderr, flush=True)
    output_dir = cfg.output_root / "vep" / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    features: dict[str, object] = {"subject": subject, "reference": ",".join(cfg.reference) or "none",
                                   "roi": ",".join(cfg.roi), "long_blocks": len(blocks["long"]),
                                   "short_blocks": len(blocks["short"])}
    mean_long = equal_block_average(blocks["long"], "corrected_mean")
    mean_short_raw = equal_block_average(blocks["short"], "epoch_mean")
    mean_short_corrected = equal_block_average(blocks["short"], "corrected_mean")
    if mean_long:
        times, average, used = mean_long
        np.savez_compressed(output_dir / "long_average.npz", times_sec=times, roi=average[:, 0], oz=average[:, 1])
        plot_averages(output_dir / f"long_erp.{plot_extension}", f"{subject}: long VEP average", times,
                      used, average, "corrected_mean", cfg.long_baseline)
        for channel, pos in (("ROI", 0), ("Oz", 1)):
            features.update(component_metrics(times, average[:, pos], f"long_{channel}"))
    if mean_short_raw:
        times, average, used = mean_short_raw
        np.savez_compressed(output_dir / "short_average_uncorrected.npz", times_sec=times,
                            roi=average[:, 0], oz=average[:, 1])
        plot_averages(output_dir / f"short_erp_uncorrected.{plot_extension}",
                      f"{subject}: short VEP trigger average, no baseline", times, used, average,
                      "epoch_mean", None)
    if mean_short_corrected:
        times, average, used = mean_short_corrected
        np.savez_compressed(output_dir / "short_average_baseline_corrected.npz", times_sec=times,
                            roi=average[:, 0], oz=average[:, 1])
        plot_averages(output_dir / f"short_erp_baseline_corrected.{plot_extension}",
                      f"{subject}: short VEP trigger average, baseline corrected", times, used,
                      average, "corrected_mean", cfg.short_baseline)
    if blocks["short"]:
        try:
            short_times, predicted, explained = fit_short_deconvolution(blocks["short"], cfg)
            features["short_deconv_roi_explained_variance"] = explained
            np.savez_compressed(output_dir / "short_deconvolved.npz", times_sec=short_times,
                                soa_ms=np.array(list(predicted)),
                                roi=np.stack([value[:, 0] for value in predicted.values()]),
                                oz=np.stack([value[:, 1] for value in predicted.values()]))
            if mean_long:
                plot_deconvolution(output_dir / f"long_vs_short_deconvolved.{plot_extension}", mean_long[0],
                                   mean_long[1], short_times, predicted)
            subject_recovery: list[dict[str, object]] = []
            for soa_ms, waveform in predicted.items():
                for channel, pos in (("ROI", 0), ("Oz", 1)):
                    metrics = component_metrics(short_times, waveform[:, pos], "short")
                    for component in COMPONENTS:
                        row: dict[str, object] = {"subject": subject, "channel": channel,
                                                  "soa_ms": soa_ms, "component": component,
                                                  "short_mean": metrics[f"short_{component}_mean"],
                                                  "short_latency_ms": metrics[f"short_{component}_latency_ms"]}
                        if mean_long:
                            long_value = features[f"long_{channel}_{component}_mean"]
                            row["long_mean"] = long_value
                            row["short_minus_long"] = float(row["short_mean"]) - float(long_value)
                            gain_valid = (abs(float(long_value)) > 1e-12
                                          and float(long_value) * float(row["short_mean"]) > 0)
                            row["gain_valid"] = gain_valid
                            row["short_long_gain"] = (float(row["short_mean"]) / float(long_value)
                                                      if gain_valid else float("nan"))
                            row["latency_difference_ms"] = (float(row["short_latency_ms"])
                                                            - float(features[f"long_{channel}_{component}_latency_ms"]))
                        subject_recovery.append(row)
                        recovery_rows.append(row)
                        if soa_ms == 100:
                            features[f"short_{channel}_{component}_100ms_mean"] = row["short_mean"]
                            features[f"gain_{channel}_{component}_100ms"] = row.get("short_long_gain", float("nan"))
            for channel in ("ROI", "Oz"):
                for component in COMPONENTS:
                    chosen = [row for row in subject_recovery if row["channel"] == channel
                              and row["component"] == component]
                    tau = fit_recovery_tau(np.array([row["soa_ms"] for row in chosen], dtype=float),
                                           np.array([row["short_mean"] for row in chosen], dtype=float))
                    features[f"recovery_tau_{channel}_{component}_ms"] = tau
            plot_recovery(output_dir / f"recovery_curve.{plot_extension}", subject_recovery)
        except (ValueError, RuntimeError) as exc:
            features["short_deconvolution_error"] = str(exc)
            print(f"  WARNING: short deconvolution failed for {subject}: {exc}", file=sys.stderr, flush=True)
    feature_rows.append(features)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=Path("VEP_Dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("analysis_results"))
    parser.add_argument("--channel-names", type=parse_names, default=DEFAULT_CHANNELS,
                        help="Eight EEG names in --eeg-columns order")
    parser.add_argument("--eeg-columns", type=parse_columns, default=tuple(range(8)))
    parser.add_argument("--marker-column", type=int, default=8)
    parser.add_argument("--reference", type=parse_names, default=("TP9", "TP10"),
                        help="Comma-separated names, or 'none' to disable software rereference")
    parser.add_argument("--roi", type=parse_names, default=("POz", "O1", "Oz", "O2"))
    parser.add_argument("--oz-channel", default="Oz")
    parser.add_argument("--long-baseline-ms", type=parse_interval, default=(-0.2, -0.02))
    parser.add_argument("--short-baseline-ms", type=parse_interval, default=(-0.05, -0.01))
    parser.add_argument("--low-hz", type=float, default=0.1)
    parser.add_argument("--high-hz", type=float, default=40.0)
    parser.add_argument("--notch-hz", type=float, default=50.0, help="0 disables the notch")
    parser.add_argument("--target-fs", type=float, default=250.0)
    parser.add_argument("--artifact-mad-multiplier", type=float, default=8.0)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    args = parser.parse_args()
    reference = () if tuple(name.lower() for name in args.reference) == ("none",) else args.reference
    cfg = Config(input_root=args.input_root, output_root=args.output_root,
                 channel_names=args.channel_names, eeg_columns=args.eeg_columns,
                 marker_column=args.marker_column, reference=reference, roi=args.roi,
                 oz_channel=args.oz_channel, long_baseline=args.long_baseline_ms,
                 short_baseline=args.short_baseline_ms, low_hz=args.low_hz,
                 high_hz=args.high_hz, notch_hz=args.notch_hz, target_fs=args.target_fs,
                 artifact_mad_multiplier=args.artifact_mad_multiplier, ridge_alpha=args.ridge_alpha)
    validate_config(cfg)
    return cfg


def main() -> int:
    cfg = parse_args()
    discovered = discover_files(cfg.input_root)
    try:
        import matplotlib
        matplotlib.use("Agg")
        plot_extension = "png"
    except ImportError:
        plot_extension = "svg"
        print("matplotlib is unavailable; writing standalone SVG plots", file=sys.stderr)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    settings = asdict(cfg)
    settings["input_root"] = str(cfg.input_root)
    settings["output_root"] = str(cfg.output_root)
    with (cfg.output_root / "vep_analysis_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    file_qc: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    for subject, modes in discovered.items():
        print(f"Analysing {subject}: {len(modes['long'])} long, {len(modes['short'])} short blocks", flush=True)
        analyse_subject(subject, modes, cfg, file_qc, feature_rows, recovery_rows, plot_extension)
    write_csv(cfg.output_root / "qc" / "recording_qc.csv", file_qc)
    write_csv(cfg.output_root / "vep" / "vep_subject_features.csv", feature_rows)
    write_csv(cfg.output_root / "vep" / "recovery_curve_by_subject.csv", recovery_rows)
    print(f"Results written to {cfg.output_root.resolve()}", flush=True)
    return 0 if any(row["status"] == "ok" for row in file_qc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
