#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-class label-switch event decoding accuracy analysis for continuous
same-frequency phase-coded SSVEP experiments.

Input data format
-----------------
Dataset/
  sub1/
    VEP_Exp2_M4_EEG_20260714_201635.mat
    VEP_Exp2_M4_States_20260714_201635.mat
    VEP_Exp2_M4_EEG_20260714_203000.mat
    VEP_Exp2_M4_States_20260714_203000.mat
  sub2/
    VEP_Exp2_M9_EEG_....mat
    VEP_Exp2_M9_States_....mat

Recorder naming convention:
  eeg_file   = f"VEP_Exp2_M{task_m}_EEG_{timestamp}.mat"
  state_file = f"VEP_Exp2_M{task_m}_States_{timestamp}.mat"

Main logic
----------
1. One EEG+States file pair is treated as one block/session.
2. The class number M is parsed from the filename: M2, M4, or M9.
3. label switch event = true_label changes from A to B.
4. For each intensity channel k, rising edges are detected from intensity_k
   independently.
5. For each test window around a switch event:
   - true label follows the original rule: determined by test_start.
   - tau1 is computed from the nearest rising edge of the configured reference
     intensity channel.
   - the same tau1 is applied to training peaks selected by the configured
     alignment mode.
6. Training windows overlapping any switch transition interval are excluded.
7. Training labels always come from true_label at the training peak time, not
   from the intensity channel that supplied the phase anchor.
8. Methods are kept close to the original swap_accuracy_analysis.py:
   Aug_LDA, Aug_FB_LDA, Aug_TRCA, Aug_FB_TRCA, Aug_eTRCA, Aug_FB_eTRCA.

Notes
-----
This is an offline analysis script. Using the true cue label to choose the
reference intensity channel for tau1 is a controlled epoch-alignment operation,
not an online decoding operation.
"""

from __future__ import annotations

import os
import re
import time as time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.linalg
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════
#  User-adjustable parameters
# ═══════════════════════════════════════════════════

ROOT_DIR = "./Dataset"
SAVE_ROOT = "./Results_LabelSwitch_Multiclass"

# Raw acquisition and decimation. The actual Fs is read from the .mat files;
# FS_RAW_DEFAULT is only a fallback.
FS_RAW_DEFAULT = 2000.0
FS_DECIMATION = 2

# Sliding test windows around label switch events. These only control the
# evaluated test offsets relative to switch_time.
WIN_LENS = [0.05]          # seconds, e.g. [0.05, 0.10, 0.50]
STEP_SIZES = [0.025]        # seconds, e.g. [0.01, 0.05]
T_BEFORE = 0.30            # seconds before switch_time
T_AFTER = 2.00             # seconds after switch_time

# Training samples whose windows overlap this transition interval are excluded.
# Keep this independent from T_BEFORE/T_AFTER so extending the test curve does
# not implicitly shrink the training set.
TRAIN_EXCLUDE_BEFORE = 0.30  # seconds before switch_time
TRAIN_EXCLUDE_AFTER = 0.50   # seconds after switch_time

# EEG channel selection: keeps the original offline analysis style.
# If your amplifier data have 9 columns, the 9th column is ignored here.
EEG_CHANS = [0, 2, 3, 4, 5, 6]
REF_CHANS = [1, 7]
USE_REREF = True
FILTER_MARGIN_S = 0.18

# Filter bank bands: same style as the original analysis script.
FB_BANDS = [
    (5.0, 95.0), (12.0, 95.0), (19.0, 95.0), (27.0, 95.0),
    (35.0, 95.0), (43.0, 95.0), (51.0, 95.0),
    (59.0, 95.0), (67.0, 95.0), (75.0, 95.0),
]

METHODS = {
    "Aug_LDA": True,
    "Aug_FB_LDA": True,
    "Aug_TRCA": True,
    "Aug_FB_TRCA": True,
    "Aug_eTRCA": True,
    "Aug_FB_eTRCA": True,
}

# Multi-class training should normally contain every class. Keep True unless
# you deliberately want partial-class folds.
REQUIRE_ALL_CLASSES = True

# Rising edge threshold for binary square-wave intensities.
INTENSITY_THRESHOLD = 0.5

# Peak alignment strategy:
#   "global_reference": match swap_accuracy_analysis.py's single
#       stimulus_intensity reference; labels still come from true_label.
#   "target_phase": use each class's own target-intensity peaks, which
#       normalizes away class phase differences and tests a different question.
PEAK_ALIGNMENT_MODE = "global_reference"
REFERENCE_INTENSITY_LABEL = 1

# Print progress every N offsets.
PRINT_EVERY_N_OFFSETS = 2


# ═══════════════════════════════════════════════════
#  Basic utilities
# ═══════════════════════════════════════════════════

@dataclass
class FilePair:
    subject: str
    task_m: int
    timestamp: str
    eeg_path: Path
    state_path: Path


@dataclass
class RawBlock:
    block_id: int
    subject: str
    task_m: int
    timestamp: str
    eeg: np.ndarray
    states: np.ndarray
    state_columns: List[str]
    fs_raw: float


def _mat_scalar(value, default=None):
    """Safely convert MATLAB scalar arrays to Python scalar."""
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return default
        return arr.item()
    except Exception:
        return default


def _decode_mat_string(x) -> str:
    """Robustly decode MATLAB cell/string values into a clean Python string."""
    if isinstance(x, str):
        return x
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.astype(str).ravel()).strip()
    if arr.dtype == object:
        if arr.size == 1:
            return _decode_mat_string(arr.ravel()[0])
        return "".join(_decode_mat_string(v) for v in arr.ravel()).strip()
    try:
        return str(arr.item()).strip()
    except Exception:
        return str(x).strip()


def parse_state_columns(raw_cols) -> List[str]:
    """Parse the saved MATLAB state_columns array."""
    arr = np.asarray(raw_cols)
    cols = []
    for item in arr.ravel():
        s = _decode_mat_string(item)
        if s:
            cols.append(s)
    return cols


def find_col_idx(columns: List[str], name: str) -> int:
    if name not in columns:
        raise KeyError(f"state_columns 中找不到列: {name}. 当前列: {columns}")
    return columns.index(name)


def find_file_pairs(subj_dir: Path) -> List[FilePair]:
    """
    Match EEG and States files by task_m and timestamp.

    Expected names:
      VEP_Exp2_M{M}_EEG_{YYYYMMDD_HHMMSS}.mat
      VEP_Exp2_M{M}_States_{YYYYMMDD_HHMMSS}.mat
    """
    pat = re.compile(r"^VEP_Exp2_M(?P<M>[249])_(?P<kind>EEG|States)_(?P<ts>\d{8}_\d{6})\.mat$")
    found: Dict[Tuple[int, str], Dict[str, Path]] = {}

    for p in sorted(subj_dir.glob("*.mat")):
        m = pat.match(p.name)
        if not m:
            continue
        task_m = int(m.group("M"))
        kind = m.group("kind")
        ts = m.group("ts")
        found.setdefault((task_m, ts), {})[kind] = p

    pairs = []
    for (task_m, ts), files in sorted(found.items(), key=lambda x: (x[0][0], x[0][1])):
        if "EEG" in files and "States" in files:
            pairs.append(
                FilePair(
                    subject=subj_dir.name,
                    task_m=task_m,
                    timestamp=ts,
                    eeg_path=files["EEG"],
                    state_path=files["States"],
                )
            )
        else:
            print(
                f"  [WARN] {subj_dir.name}: M{task_m} {ts} 文件不成对，"
                f"已有 {list(files.keys())}，跳过。"
            )
    return pairs


def load_file_pair(pair: FilePair, block_id: int) -> RawBlock:
    eeg_mat = sio.loadmat(str(pair.eeg_path))
    state_mat = sio.loadmat(str(pair.state_path))

    if "data" not in eeg_mat:
        raise KeyError(f"{pair.eeg_path} 中找不到 data")
    if "states" not in state_mat:
        raise KeyError(f"{pair.state_path} 中找不到 states")
    if "state_columns" not in state_mat:
        raise KeyError(f"{pair.state_path} 中找不到 state_columns")

    eeg = np.asarray(eeg_mat["data"], dtype=np.float64)
    states = np.asarray(state_mat["states"], dtype=np.float64)
    fs_eeg = float(_mat_scalar(eeg_mat.get("Fs", FS_RAW_DEFAULT), FS_RAW_DEFAULT))
    fs_state = float(_mat_scalar(state_mat.get("Fs", fs_eeg), fs_eeg))
    state_columns = parse_state_columns(state_mat["state_columns"])

    if abs(fs_eeg - fs_state) > 1e-6:
        print(
            f"  [WARN] {pair.subject} M{pair.task_m} {pair.timestamp}: "
            f"EEG Fs={fs_eeg}, States Fs={fs_state}; 使用 EEG Fs。"
        )

    # Align lengths conservatively.
    n = min(eeg.shape[0], states.shape[0])
    if eeg.shape[0] != states.shape[0]:
        print(
            f"  [WARN] {pair.subject} M{pair.task_m} {pair.timestamp}: "
            f"EEG length={eeg.shape[0]}, States length={states.shape[0]}; 截断到 {n}。"
        )
    eeg = eeg[:n]
    states = states[:n]

    return RawBlock(
        block_id=block_id,
        subject=pair.subject,
        task_m=pair.task_m,
        timestamp=pair.timestamp,
        eeg=eeg,
        states=states,
        state_columns=state_columns,
        fs_raw=fs_eeg,
    )


# ═══════════════════════════════════════════════════
#  Signal processing
# ═══════════════════════════════════════════════════

def apply_notch(data, fs, freq=50.0, Q=30.0, axis=-1):
    """Apply a one-pass causal notch filter without future-sample leakage."""
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, Q)
    sos = tf2sos(b, a)
    return sosfilt(sos, data, axis=axis)


def apply_bandpass(data, fs, low, high, order=4, axis=-1):
    """Apply a one-pass causal Butterworth bandpass filter."""
    nyq = 0.5 * fs
    if high >= nyq:
        high = nyq * 0.95
    if low <= 0 or low >= high:
        raise ValueError(f"Invalid bandpass range: low={low}, high={high}, fs={fs}")
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, data, axis=axis)


def find_rising_edges(signal, threshold=0.5):
    below = signal[:-1] < threshold
    above = signal[1:] >= threshold
    return np.where(below & above)[0] + 1


def find_nearest_peak(peaks_positions, target_pos):
    """Find the nearest peak to target_pos. Kept intentionally simple."""
    if len(peaks_positions) == 0:
        return None, None
    dists = np.abs(peaks_positions - target_pos)
    idx = np.argmin(dists)
    return int(peaks_positions[idx]), int(dists[idx])


def preprocess_block(block: RawBlock, fs_dec: int, use_reref=True):
    """
    Preprocess one raw block.

    Returns
    -------
    eeg_wide : (T', n_ch)
    eeg_fb   : list of (T', n_ch)
    states_ds: (T', n_state_cols)
    fs       : decimated sampling rate
    """
    fs_raw = float(block.fs_raw)
    fs = fs_raw / fs_dec
    eeg_raw = np.asarray(block.eeg, dtype=np.float64)

    min_required_col = max(max(EEG_CHANS), max(REF_CHANS)) + 1
    if eeg_raw.shape[1] < min_required_col:
        raise ValueError(
            f"block {block.block_id}: EEG 列数={eeg_raw.shape[1]}，"
            f"不足以使用 EEG_CHANS={EEG_CHANS}, REF_CHANS={REF_CHANS}"
        )

    eeg = eeg_raw.copy()
    if use_reref:
        ref = np.mean(eeg[:, REF_CHANS], axis=1, keepdims=True)
        eeg = eeg - ref
    eeg = eeg[:, EEG_CHANS]

    eeg = apply_notch(eeg, fs_raw, freq=50.0, axis=0)
    eeg_wide_full = apply_bandpass(eeg, fs_raw, 3.0, 100.0, axis=0)
    eeg_wide = eeg_wide_full[::fs_dec, :]

    eeg_fb = []
    for low, high in FB_BANDS:
        band = apply_bandpass(eeg, fs_raw, low, high, axis=0)
        eeg_fb.append(band[::fs_dec, :])

    states_ds = block.states[::fs_dec, :]
    return eeg_wide, eeg_fb, states_ds, fs


# ═══════════════════════════════════════════════════
#  TRCA / eTRCA models
# ═══════════════════════════════════════════════════

def _safe_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size == 0:
        return -np.inf
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -np.inf
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else -np.inf


class TRCA:
    def __init__(self):
        self.W = []
        self.templates = []

    def fit(self, train_data_list):
        """train_data_list: list of (n_trials, n_ch, n_samples) per class."""
        self.templates = []
        self.W = []
        for data in train_data_list:
            n_trials, n_ch, _ = data.shape
            self.templates.append(np.mean(data, axis=0))
            dc = data - data.mean(axis=2, keepdims=True)
            SX = dc.sum(axis=0)
            S = SX @ SX.T + np.eye(n_ch) * 1e-9
            UX = dc.transpose(1, 2, 0).reshape(n_ch, -1, order="F")
            Q = UX @ UX.T + np.eye(n_ch) * 1e-9
            try:
                _, W = scipy.linalg.eigh(S, Q)
                w = W[:, -1]
                self.W.append(w / (np.linalg.norm(w) + 1e-12))
            except Exception:
                self.W.append(np.ones(n_ch) / np.sqrt(n_ch))

    def predict_corr(self, test_data):
        corrs = []
        for i, w in enumerate(self.W):
            corrs.append(_safe_corr(w @ test_data, w @ self.templates[i]))
        return np.array(corrs)

    def predict(self, test_data):
        return int(np.argmax(self.predict_corr(test_data)))


class eTRCA:
    def __init__(self):
        self.W_stack = None
        self.template_feats = []

    def fit(self, train_data_list):
        W_list = []
        templates = []
        for data in train_data_list:
            n_trials, n_ch, _ = data.shape
            templates.append(np.mean(data, axis=0))
            dc = data - data.mean(axis=2, keepdims=True)
            SX = dc.sum(axis=0)
            S = SX @ SX.T + np.eye(n_ch) * 1e-9
            UX = dc.transpose(1, 2, 0).reshape(n_ch, -1, order="F")
            Q = UX @ UX.T + np.eye(n_ch, dtype=float) * 1e-9
            try:
                _, W = scipy.linalg.eigh(S, Q)
                w = W[:, -1]
                W_list.append(w / (np.linalg.norm(w) + 1e-12))
            except Exception:
                W_list.append(np.ones(n_ch) / np.sqrt(n_ch))
        self.W_stack = np.stack(W_list, axis=1)
        self.template_feats = [(self.W_stack.T @ t).reshape(-1) for t in templates]

    def predict_corr(self, test_data):
        feat = (self.W_stack.T @ test_data).reshape(-1)
        return np.array([_safe_corr(feat, tf) for tf in self.template_feats])

    def predict(self, test_data):
        return int(np.argmax(self.predict_corr(test_data)))


# ═══════════════════════════════════════════════════
#  Event and peak construction
# ═══════════════════════════════════════════════════

def build_switch_events_for_block(
    block_id: int,
    true_label_ds: np.ndarray,
    task_m: int,
) -> List[dict]:
    """Build cue/label switch events from decimated true_label."""
    labels = np.asarray(true_label_ds).astype(int)
    valid = (labels >= 1) & (labels <= task_m)
    events = []
    for i in range(1, len(labels)):
        if not (valid[i - 1] and valid[i]):
            continue
        if labels[i] != labels[i - 1]:
            events.append(
                {
                    "block": block_id,
                    "switch_time": int(i),
                    "before_label": int(labels[i - 1]),
                    "after_label": int(labels[i]),
                }
            )
    return events


def build_peaks_for_block(
    block_id: int,
    states_ds: np.ndarray,
    state_columns: List[str],
    task_m: int,
    true_label_ds: np.ndarray,
    alignment_mode: str,
    reference_intensity_label: int,
) -> dict:
    """
    Detect rising edges for intensity_1 ... intensity_M independently.

    Returns both per-intensity-channel peaks for test phase alignment and a
    merged training peak table. Merged labels are always true_label values.
    """
    if alignment_mode not in {"global_reference", "target_phase"}:
        raise ValueError(f"Unknown PEAK_ALIGNMENT_MODE: {alignment_mode}")
    if not 1 <= int(reference_intensity_label) <= task_m:
        raise ValueError(
            f"REFERENCE_INTENSITY_LABEL={reference_intensity_label} is outside 1..{task_m}"
        )

    positions_by_intensity: Dict[int, np.ndarray] = {}
    all_pos = []
    all_lab = []
    all_anchor = []
    true_labels = np.asarray(true_label_ds).astype(int)

    for intensity_label in range(1, task_m + 1):
        col_name = f"intensity_{intensity_label}"
        col_idx = find_col_idx(state_columns, col_name)
        intensity = states_ds[:, col_idx]
        pos = find_rising_edges(intensity, threshold=INTENSITY_THRESHOLD).astype(int)
        positions_by_intensity[intensity_label] = pos
        if len(pos) == 0:
            continue

        valid = (pos >= 0) & (pos < len(true_labels))
        pos = pos[valid]
        peak_true_labels = true_labels[pos]
        valid_labels = (peak_true_labels >= 1) & (peak_true_labels <= task_m)
        if alignment_mode == "global_reference":
            valid_labels &= intensity_label == int(reference_intensity_label)
        elif alignment_mode == "target_phase":
            valid_labels &= peak_true_labels == intensity_label
        pos = pos[valid_labels]
        peak_true_labels = peak_true_labels[valid_labels]
        if len(pos) > 0:
            all_pos.append(pos)
            all_lab.append(peak_true_labels.astype(int))
            all_anchor.append(np.full(len(pos), intensity_label, dtype=int))

    if all_pos:
        positions = np.concatenate(all_pos)
        labels = np.concatenate(all_lab)
        anchor_labels = np.concatenate(all_anchor)
        order = np.argsort(positions, kind="stable")
        positions = positions[order]
        labels = labels[order]
        anchor_labels = anchor_labels[order]
    else:
        positions = np.array([], dtype=int)
        labels = np.array([], dtype=int)
        anchor_labels = np.array([], dtype=int)

    return {
        "block": block_id,
        "positions_by_intensity": positions_by_intensity,
        "positions": positions,
        "labels": labels,
        "anchor_labels": anchor_labels,
    }


# ═══════════════════════════════════════════════════
#  Training data construction
# ═══════════════════════════════════════════════════

def prepare_train_data(
    eeg_blocks: Dict[int, np.ndarray],
    peaks_by_block: Dict[int, dict],
    win_len_samples: int,
    tau1: int,
    fs: float,
    margin_s: float,
    exclude_before_s: float,
    exclude_after_s: float,
    all_switch_events: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training samples from selected intensity peaks using one tau1.

    Important:
      - tau1 is computed from the current test window.
      - selected training peaks are then cut with the same tau1.
      - labels come from true_label at each training peak time.
      - any training window overlapping a configured switch transition
        interval is excluded.
    """
    margin_samples = int(round(margin_s * fs))
    before_samples = int(round(exclude_before_s * fs))
    after_samples = int(round(exclude_after_s * fs))

    exclude_ranges_by_block: Dict[int, List[Tuple[int, int]]] = {}
    for se in all_switch_events:
        blk = int(se["block"])
        st = int(se["switch_time"]) - before_samples
        ed = int(se["switch_time"]) + after_samples
        exclude_ranges_by_block.setdefault(blk, []).append((st, ed))

    X_list = []
    Y_list = []

    for block_id, eeg in eeg_blocks.items():
        T = eeg.shape[0]
        peaks = peaks_by_block[block_id]
        ranges = exclude_ranges_by_block.get(block_id, [])

        for peak_pos, peak_label in zip(peaks["positions"], peaks["labels"]):
            train_start = int(peak_pos) - int(tau1)
            train_end = train_start + win_len_samples

            # Keep the original boundary/margin style.
            if train_start < margin_samples or train_end > T:
                continue
            if int(peak_label) < 1:
                continue

            # Exclude the whole training window, not only its phase anchor.
            if any(train_start < e and s < train_end for s, e in ranges):
                continue

            seg = eeg[train_start:train_end, :]
            if seg.shape[0] != win_len_samples:
                continue
            X_list.append(seg.T)
            Y_list.append(int(peak_label) - 1)  # true_label 1..M -> 0..M-1

    if len(X_list) == 0:
        return np.empty((0,)), np.empty((0,), dtype=int)
    return np.stack(X_list), np.asarray(Y_list, dtype=int)


def has_all_classes(Y: np.ndarray, task_m: int) -> bool:
    if Y.size == 0:
        return False
    classes = set(np.unique(Y).astype(int).tolist())
    return all(c in classes for c in range(task_m))


# ═══════════════════════════════════════════════════
#  Model training and prediction
# ═══════════════════════════════════════════════════

def train_and_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_train_fb: List[np.ndarray],
    test_sample: np.ndarray,
    test_sample_fb: np.ndarray,
    methods_config: dict,
    task_m: int,
) -> Dict[str, int]:
    """
    Train all enabled methods and predict one M-class test sample.

    Returns method -> predicted label in 0..M-1, or -1 if failed.
    """
    preds = {}
    n_samples = X_train.shape[0] if X_train.ndim == 3 else 0
    active_classes = np.unique(Y_train).astype(int) if n_samples > 0 else np.array([])

    if REQUIRE_ALL_CLASSES:
        can_fit_single = n_samples > task_m and has_all_classes(Y_train, task_m)
    else:
        can_fit_single = n_samples > len(active_classes) and len(active_classes) >= 2

    def group_by_label(X, Y):
        # Return groups in fixed 0..M-1 order. This keeps TRCA/eTRCA output
        # indices identical to class labels.
        return [X[Y == c] for c in range(task_m)]

    # Aug_LDA
    if methods_config.get("Aug_LDA"):
        if can_fit_single:
            try:
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                lda.fit(X_train.reshape(n_samples, -1), Y_train)
                preds["Aug_LDA"] = int(lda.predict(test_sample.flatten().reshape(1, -1))[0])
            except Exception:
                preds["Aug_LDA"] = -1
        else:
            preds["Aug_LDA"] = -1

    # Aug_TRCA
    if methods_config.get("Aug_TRCA"):
        if can_fit_single:
            try:
                grouped = group_by_label(X_train, Y_train)
                if any(g.shape[0] == 0 for g in grouped):
                    preds["Aug_TRCA"] = -1
                else:
                    m = TRCA()
                    m.fit(grouped)
                    preds["Aug_TRCA"] = m.predict(test_sample)
            except Exception:
                preds["Aug_TRCA"] = -1
        else:
            preds["Aug_TRCA"] = -1

    # Aug_eTRCA
    if methods_config.get("Aug_eTRCA"):
        if can_fit_single:
            try:
                grouped = group_by_label(X_train, Y_train)
                if any(g.shape[0] == 0 for g in grouped):
                    preds["Aug_eTRCA"] = -1
                else:
                    m = eTRCA()
                    m.fit(grouped)
                    preds["Aug_eTRCA"] = m.predict(test_sample)
            except Exception:
                preds["Aug_eTRCA"] = -1
        else:
            preds["Aug_eTRCA"] = -1

    # Filter-bank methods
    n_bands = len(X_train_fb)

    # Aug_FB_LDA: per-band LDA -> decision_function stack -> final LDA.
    if methods_config.get("Aug_FB_LDA"):
        try:
            if not can_fit_single:
                raise RuntimeError("training set not fit-ready")
            sub_models = []
            stacked_feats = []
            target_y = None
            for band_idx in range(n_bands):
                Xb = X_train_fb[band_idx]
                if Xb.ndim != 3 or Xb.shape[0] == 0:
                    continue
                if REQUIRE_ALL_CLASSES and not has_all_classes(Y_train, task_m):
                    continue
                lda_b = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                lda_b.fit(Xb.reshape(Xb.shape[0], -1), Y_train)
                sub_models.append((band_idx, lda_b))
                feat = lda_b.decision_function(Xb.reshape(Xb.shape[0], -1))
                if feat.ndim == 1:
                    feat = feat.reshape(-1, 1)
                stacked_feats.append(feat)
                if target_y is None:
                    target_y = Y_train

            if len(stacked_feats) == 0 or target_y is None:
                preds["Aug_FB_LDA"] = -1
            else:
                final_X = np.concatenate(stacked_feats, axis=1)
                final_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                final_lda.fit(final_X, target_y)

                test_feats = []
                for band_idx, lda_b in sub_models:
                    tb = test_sample_fb[band_idx].flatten().reshape(1, -1)
                    val = lda_b.decision_function(tb)
                    if val.ndim == 1:
                        val = val.reshape(-1, 1)
                    test_feats.append(val)
                test_final = np.concatenate(test_feats, axis=1)
                preds["Aug_FB_LDA"] = int(final_lda.predict(test_final)[0])
        except Exception:
            preds["Aug_FB_LDA"] = -1

    # Aug_FB_TRCA: per-band TRCA -> weighted sum of correlations.
    if methods_config.get("Aug_FB_TRCA"):
        try:
            if not can_fit_single:
                raise RuntimeError("training set not fit-ready")
            sum_corrs = None
            for band_idx in range(n_bands):
                Xb = X_train_fb[band_idx]
                if Xb.ndim != 3 or Xb.shape[0] == 0:
                    continue
                grouped = group_by_label(Xb, Y_train)
                if any(g.shape[0] == 0 for g in grouped):
                    continue
                m = TRCA()
                m.fit(grouped)
                weight = ((band_idx + 1) ** (-1.25)) + 0.25
                corrs = m.predict_corr(test_sample_fb[band_idx])
                sum_corrs = weight * corrs if sum_corrs is None else sum_corrs + weight * corrs
            preds["Aug_FB_TRCA"] = int(np.argmax(sum_corrs)) if sum_corrs is not None else -1
        except Exception:
            preds["Aug_FB_TRCA"] = -1

    # Aug_FB_eTRCA: per-band eTRCA -> weighted sum of correlations.
    if methods_config.get("Aug_FB_eTRCA"):
        try:
            if not can_fit_single:
                raise RuntimeError("training set not fit-ready")
            sum_corrs = None
            for band_idx in range(n_bands):
                Xb = X_train_fb[band_idx]
                if Xb.ndim != 3 or Xb.shape[0] == 0:
                    continue
                grouped = group_by_label(Xb, Y_train)
                if any(g.shape[0] == 0 for g in grouped):
                    continue
                m = eTRCA()
                m.fit(grouped)
                weight = ((band_idx + 1) ** (-1.25)) + 0.25
                corrs = m.predict_corr(test_sample_fb[band_idx])
                sum_corrs = weight * corrs if sum_corrs is None else sum_corrs + weight * corrs
            preds["Aug_FB_eTRCA"] = int(np.argmax(sum_corrs)) if sum_corrs is not None else -1
        except Exception:
            preds["Aug_FB_eTRCA"] = -1

    return preds


# ═══════════════════════════════════════════════════
#  Analysis core
# ═══════════════════════════════════════════════════

def process_task_blocks(
    subject: str,
    task_m: int,
    raw_blocks: List[RawBlock],
    save_dir: Path,
    methods_config: dict,
    win_lens: List[float],
    step_size: float,
    t_before: float,
    t_after: float,
):
    """Analyze all blocks of one subject and one M condition."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fs_list = [b.fs_raw / FS_DECIMATION for b in raw_blocks]
    if max(fs_list) - min(fs_list) > 1e-6:
        raise ValueError(f"{subject} M{task_m}: 多个文件降采样 Fs 不一致: {fs_list}")
    fs = fs_list[0]

    print(f"\n{'=' * 72}")
    print(f"Subject={subject} | M={task_m} | blocks={len(raw_blocks)} | Fs_ds={fs:.1f} Hz")
    print(
        f"Test scan: -{t_before:.3f}s to +{t_after:.3f}s | "
        f"Train exclude: -{TRAIN_EXCLUDE_BEFORE:.3f}s to +{TRAIN_EXCLUDE_AFTER:.3f}s"
    )
    print(f"{'=' * 72}")

    # 1) Preprocess all blocks, build peaks and switch events.
    eeg_wide_blocks: Dict[int, np.ndarray] = {}
    eeg_fb_blocks: Dict[int, List[np.ndarray]] = {}
    peaks_by_block: Dict[int, dict] = {}
    all_switch_events: List[dict] = []

    for block in raw_blocks:
        true_label_col = find_col_idx(block.state_columns, "true_label")
        for k in range(1, task_m + 1):
            find_col_idx(block.state_columns, f"intensity_{k}")

        eeg_w, eeg_fb, states_ds, fs_ds = preprocess_block(block, FS_DECIMATION, USE_REREF)
        eeg_wide_blocks[block.block_id] = eeg_w
        eeg_fb_blocks[block.block_id] = eeg_fb

        true_label_ds = states_ds[:, true_label_col]
        switches = build_switch_events_for_block(block.block_id, true_label_ds, task_m)
        all_switch_events.extend(switches)

        peaks = build_peaks_for_block(
            block.block_id,
            states_ds,
            block.state_columns,
            task_m,
            true_label_ds,
            PEAK_ALIGNMENT_MODE,
            REFERENCE_INTENSITY_LABEL,
        )
        peaks_by_block[block.block_id] = peaks

        peak_counts = [len(peaks["positions_by_intensity"][k]) for k in range(1, task_m + 1)]
        eligible_peak_label_counts = np.bincount(peaks["labels"], minlength=task_m + 1)[1:]
        label_counts = np.bincount(
            np.clip(true_label_ds.astype(int), 0, task_m), minlength=task_m + 1
        )[1:]
        print(
            f"  block {block.block_id:02d} | {block.timestamp} | "
            f"EEG_ds={eeg_w.shape} | switches={len(switches)} | "
            f"peaks/intensity={peak_counts} | selected_train_peaks={eligible_peak_label_counts.tolist()} | "
            f"label_samples={label_counts.tolist()}"
        )

    if len(all_switch_events) == 0:
        print(f"  [SKIP] {subject} M{task_m}: 没有 label switch event。")
        return None, None

    # Build transition matrix only for debug output.
    trans_mat = np.zeros((task_m, task_m), dtype=int)
    for se in all_switch_events:
        trans_mat[se["before_label"] - 1, se["after_label"] - 1] += 1
    print(f"  Total switch events: {len(all_switch_events)}")
    print("  Transition count matrix (rows=before, cols=after):")
    for r in range(task_m):
        print(f"    {r + 1}: {trans_mat[r].tolist()}")

    # 2) Sliding-window LOOCV-like analysis.
    all_results = {}
    active_methods = [k for k, v in methods_config.items() if v]
    train_size_records = []
    skip_reasons_total = {
        "test_boundary": 0,
        "missing_true_label_peaks": 0,
        "empty_or_incomplete_train": 0,
        "fb_train_failed": 0,
    }

    for win_len in win_lens:
        win_samples = int(round(win_len * fs))
        t_offsets_s = np.arange(-t_before, t_after + 1e-12, step_size)
        t_offsets_smp = np.round(t_offsets_s * fs).astype(int)
        n_offsets = len(t_offsets_s)
        n_switches = len(all_switch_events)

        print(
            f"\n  Window={win_len:.3f}s ({win_samples} samples) | "
            f"step={step_size:.3f}s | offsets={n_offsets} | switches={n_switches}"
        )

        acc_curves = {m: np.full(n_offsets, np.nan, dtype=float) for m in active_methods}
        valid_counts = np.zeros(n_offsets, dtype=int)
        correct_counts_by_method = {m: np.zeros(n_offsets, dtype=int) for m in active_methods}
        train_counts_by_offset = [[] for _ in range(n_offsets)]

        for t_idx, (t_off_s, t_off_smp) in enumerate(zip(t_offsets_s, t_offsets_smp)):
            t_step_start = time_module.time()
            correct_counts = {m: 0 for m in active_methods}
            valid_tests = 0
            skip_reasons = {k: 0 for k in skip_reasons_total}

            for se in all_switch_events:
                blk = int(se["block"])
                eeg_w = eeg_wide_blocks[blk]
                test_start = int(se["switch_time"]) + int(t_off_smp)
                test_end = test_start + win_samples

                if test_start < 0 or test_end > eeg_w.shape[0]:
                    skip_reasons["test_boundary"] += 1
                    continue

                # Original rule: use test_start to determine the test label.
                if test_start < int(se["switch_time"]):
                    true_label_1b = int(se["before_label"])
                else:
                    true_label_1b = int(se["after_label"])
                true_label_0b = true_label_1b - 1

                # tau1: from the configured reference intensity channel.
                if PEAK_ALIGNMENT_MODE == "global_reference":
                    tau1_intensity_label = int(REFERENCE_INTENSITY_LABEL)
                elif PEAK_ALIGNMENT_MODE == "target_phase":
                    tau1_intensity_label = true_label_1b
                else:
                    raise ValueError(f"Unknown PEAK_ALIGNMENT_MODE: {PEAK_ALIGNMENT_MODE}")
                test_center = (test_start + test_end) // 2
                label_peaks = peaks_by_block[blk]["positions_by_intensity"].get(
                    tau1_intensity_label,
                    np.array([]),
                )
                peak_pos, peak_dist = find_nearest_peak(label_peaks, test_center)
                if peak_pos is None:
                    skip_reasons["missing_true_label_peaks"] += 1
                    continue
                tau1 = int(peak_pos) - int(test_start)

                # Test sample.
                test_sample = eeg_w[test_start:test_end, :].T
                test_sample_fb = np.stack(
                    [
                        eeg_fb_blocks[blk][b][test_start:test_end, :].T
                        for b in range(len(FB_BANDS))
                    ],
                    axis=0,
                )

                # Build broadband training data using the selected alignment strategy and this tau1.
                X_train, Y_train = prepare_train_data(
                    eeg_wide_blocks,
                    peaks_by_block,
                    win_samples,
                    tau1,
                    fs,
                    FILTER_MARGIN_S,
                    TRAIN_EXCLUDE_BEFORE,
                    TRAIN_EXCLUDE_AFTER,
                    all_switch_events,
                )

                if X_train.ndim != 3 or X_train.shape[0] == 0:
                    skip_reasons["empty_or_incomplete_train"] += 1
                    continue
                if REQUIRE_ALL_CLASSES and not has_all_classes(Y_train, task_m):
                    skip_reasons["empty_or_incomplete_train"] += 1
                    continue
                if (not REQUIRE_ALL_CLASSES) and len(np.unique(Y_train)) < 2:
                    skip_reasons["empty_or_incomplete_train"] += 1
                    continue

                # Build FB training data with the exact same peak table and tau1.
                X_train_fb = []
                fb_ok = True
                for b_idx in range(len(FB_BANDS)):
                    fb_blocks_b = {bid: eeg_fb_blocks[bid][b_idx] for bid in eeg_fb_blocks}
                    Xb, _ = prepare_train_data(
                        fb_blocks_b,
                        peaks_by_block,
                        win_samples,
                        tau1,
                        fs,
                        FILTER_MARGIN_S,
                        TRAIN_EXCLUDE_BEFORE,
                        TRAIN_EXCLUDE_AFTER,
                        all_switch_events,
                    )
                    if Xb.ndim != 3 or Xb.shape[0] != X_train.shape[0]:
                        fb_ok = False
                    X_train_fb.append(Xb)
                if not fb_ok:
                    skip_reasons["fb_train_failed"] += 1
                    continue

                train_size_records.append(int(X_train.shape[0]))
                train_counts_by_offset[t_idx].append(int(X_train.shape[0]))

                preds = train_and_predict(
                    X_train,
                    Y_train,
                    X_train_fb,
                    test_sample,
                    test_sample_fb,
                    methods_config,
                    task_m,
                )

                valid_tests += 1
                for m_name, pred in preds.items():
                    if pred == true_label_0b:
                        correct_counts[m_name] += 1

            # Save accuracy at this offset.
            valid_counts[t_idx] = valid_tests
            for m_name in active_methods:
                correct_counts_by_method[m_name][t_idx] = correct_counts[m_name]
                acc_curves[m_name][t_idx] = (
                    correct_counts[m_name] / valid_tests if valid_tests > 0 else np.nan
                )

            for k in skip_reasons_total:
                skip_reasons_total[k] += skip_reasons[k]

            elapsed = time_module.time() - t_step_start
            if t_idx % PRINT_EVERY_N_OFFSETS == 0 or t_idx == n_offsets - 1:
                acc_str = " ".join(
                    [
                        f"{m}:{acc_curves[m][t_idx] * 100:5.1f}%"
                        if np.isfinite(acc_curves[m][t_idx]) else f"{m}:  nan"
                        for m in active_methods
                    ]
                )
                if train_counts_by_offset[t_idx]:
                    tr_mean = np.mean(train_counts_by_offset[t_idx])
                    tr_txt = f"train_mean={tr_mean:.1f}"
                else:
                    tr_txt = "train_mean=NA"
                print(
                    f"    t={t_off_s:+.3f}s | valid={valid_tests}/{n_switches} | "
                    f"{tr_txt} | {acc_str} | {elapsed:.3f}s"
                )

        all_results[win_len] = {
            "t_offsets_s": t_offsets_s,
            "acc_curves": acc_curves,
            "valid_counts": valid_counts,
            "correct_counts": correct_counts_by_method,
            "task_m": task_m,
            "chance_level": 1.0 / task_m,
            "win_len_s": win_len,
            "step_size_s": step_size,
        }

    train_stats = None
    if train_size_records:
        train_stats = {
            "min": int(np.min(train_size_records)),
            "max": int(np.max(train_size_records)),
            "mean": float(np.mean(train_size_records)),
            "median": float(np.median(train_size_records)),
            "count": int(len(train_size_records)),
        }
        print(
            f"\n  Train size stats: min={train_stats['min']}, "
            f"max={train_stats['max']}, mean={train_stats['mean']:.1f}, "
            f"median={train_stats['median']:.1f}, folds={train_stats['count']}"
        )
    print(f"  Skip summary: {skip_reasons_total}")

    # Save and plot.
    win_tag = "-".join(f"{int(w * 1000)}ms" for w in win_lens)
    step_ms = int(round(step_size * 1000))
    npy_path = save_dir / f"LabelSwitch_M{task_m}_step{step_ms}ms_win{win_tag}.npy"
    np.save(str(npy_path), all_results)
    print(f"  Saved results: {npy_path}")

    stats_path = save_dir / f"LabelSwitch_M{task_m}_step{step_ms}ms_win{win_tag}_stats.txt"
    save_stats_text(
        stats_path,
        subject,
        task_m,
        raw_blocks,
        all_switch_events,
        peaks_by_block,
        all_results,
        train_stats,
        skip_reasons_total,
    )
    plot_task_results(all_results, subject, task_m, save_dir)

    return all_results, train_stats


def save_stats_text(
    path: Path,
    subject: str,
    task_m: int,
    raw_blocks: List[RawBlock],
    all_switch_events: List[dict],
    peaks_by_block: Dict[int, dict],
    all_results: dict,
    train_stats: Optional[dict],
    skip_summary: dict,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Task M: {task_m}\n")
        f.write(f"Blocks: {len(raw_blocks)}\n")
        f.write(f"Switch events: {len(all_switch_events)}\n")
        f.write(f"Chance level: {100 / task_m:.2f}%\n")
        f.write("\nBlock files:\n")
        for b in raw_blocks:
            f.write(f"  block {b.block_id}: {b.timestamp}, Fs={b.fs_raw}, EEG={b.eeg.shape}, States={b.states.shape}\n")
        f.write(f"\nTest scan window: -{T_BEFORE:.3f}s to +{T_AFTER:.3f}s\n")
        f.write(
            f"Training exclude window: -{TRAIN_EXCLUDE_BEFORE:.3f}s "
            f"to +{TRAIN_EXCLUDE_AFTER:.3f}s\n"
        )
        f.write(f"\nPeak alignment mode: {PEAK_ALIGNMENT_MODE}\n")
        f.write(f"Reference intensity label: {REFERENCE_INTENSITY_LABEL}\n")
        f.write("\nPeaks per intensity channel and selected training peaks:\n")
        for b in raw_blocks:
            peaks = peaks_by_block[b.block_id]
            intensity_counts = [len(peaks["positions_by_intensity"][k]) for k in range(1, task_m + 1)]
            selected_counts = np.bincount(peaks["labels"], minlength=task_m + 1)[1:]
            f.write(
                f"  block {b.block_id}: intensity={intensity_counts}, "
                f"selected_train={selected_counts.tolist()}\n"
            )
        f.write("\nTrain size stats:\n")
        f.write(f"  {train_stats}\n")
        f.write("\nSkip summary:\n")
        f.write(f"  {skip_summary}\n")
        f.write("\nResults summary by window/method:\n")
        for win_len, res in all_results.items():
            f.write(f"\n  Window {win_len}s\n")
            valid = res["valid_counts"]
            f.write(f"    valid_tests: min={np.nanmin(valid)}, max={np.nanmax(valid)}, mean={np.nanmean(valid):.1f}\n")
            for m, acc in res["acc_curves"].items():
                f.write(f"    {m}: mean_acc={np.nanmean(acc) * 100:.2f}%, max_acc={np.nanmax(acc) * 100:.2f}%\n")
    print(f"  Saved stats: {path}")


# ═══════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════

def plot_task_results(all_results: dict, subject: str, task_m: int, save_dir: Path):
    colors_by_method = {
        "Aug_LDA": "#1f77b4",
        "Aug_FB_LDA": "#ff7f0e",
        "Aug_TRCA": "#2ca02c",
        "Aug_FB_TRCA": "#d62728",
        "Aug_eTRCA": "#9467bd",
        "Aug_FB_eTRCA": "#8c564b",
    }
    chance = 100.0 / task_m

    for win_len, res in all_results.items():
        t_ms = res["t_offsets_s"] * 1000.0
        fig, ax = plt.subplots(figsize=(12, 6))
        for m_name, acc in res["acc_curves"].items():
            style = "-" if m_name == "Aug_FB_LDA" else "--"
            lw = 2.0 if m_name == "Aug_FB_LDA" else 1.5
            alpha = 1.0 if m_name == "Aug_FB_LDA" else 0.75
            ax.plot(
                t_ms,
                acc * 100.0,
                color=colors_by_method.get(m_name, "gray"),
                linestyle=style,
                linewidth=lw,
                alpha=alpha,
                label=m_name,
            )
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.6, label="label switch")
        ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.8, label=f"chance={chance:.1f}%")
        ax.set_xlabel("Time relative to label switch (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{subject} M{task_m} Label-Switch Decoding Accuracy (win={win_len}s)")
        ax.set_ylim([-5, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig_path = save_dir / f"LabelSwitch_{subject}_M{task_m}_win{int(win_len * 1000)}ms.png"
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")

    # One panel per method, lines are different window lengths.
    sample_res = list(all_results.values())[0]
    methods = list(sample_res["acc_curves"].keys())
    n_methods = len(methods)
    n_cols = 3
    n_rows = int(np.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    win_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_results)))

    for m_idx, m_name in enumerate(methods):
        ax = axes[m_idx]
        for w_idx, (wl, res) in enumerate(sorted(all_results.items())):
            ax.plot(
                res["t_offsets_s"] * 1000.0,
                res["acc_curves"][m_name] * 100.0,
                color=win_colors[w_idx],
                linewidth=1.8,
                label=f"{int(wl * 1000)}ms",
            )
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.6)
        ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.8)
        ax.set_title(m_name)
        ax.set_ylim([-5, 105])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[n_methods:]:
        ax.axis("off")
    fig.suptitle(f"{subject} M{task_m} Label-Switch Decoding: all windows", fontsize=14)
    fig.tight_layout()
    fig_path = save_dir / f"LabelSwitch_{subject}_M{task_m}_all_windows.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    print(f"  Saved combined figure: {fig_path}")


def plot_group_average(group_results: Dict[str, dict], task_m: int, save_dir: Path):
    """Average curves across subjects for the same M condition."""
    if not group_results:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    first_subject = next(iter(group_results))
    sample_result = group_results[first_subject]
    win_lens = sorted(sample_result.keys())
    chance = 100.0 / task_m

    colors_by_method = {
        "Aug_LDA": "#1f77b4",
        "Aug_FB_LDA": "#ff7f0e",
        "Aug_TRCA": "#2ca02c",
        "Aug_FB_TRCA": "#d62728",
        "Aug_eTRCA": "#9467bd",
        "Aug_FB_eTRCA": "#8c564b",
    }

    for wl in win_lens:
        methods = list(sample_result[wl]["acc_curves"].keys())
        t_ms = sample_result[wl]["t_offsets_s"] * 1000.0
        fig, ax = plt.subplots(figsize=(12, 6))
        for m_name in methods:
            curves = []
            for subj, res in group_results.items():
                if wl in res and m_name in res[wl]["acc_curves"]:
                    curves.append(res[wl]["acc_curves"][m_name])
            if not curves:
                continue
            mean_acc = np.nanmean(np.stack(curves, axis=0), axis=0)
            style = "-" if m_name == "Aug_FB_LDA" else "--"
            ax.plot(
                t_ms,
                mean_acc * 100.0,
                color=colors_by_method.get(m_name, "gray"),
                linestyle=style,
                linewidth=2.0,
                label=m_name,
            )
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.6, label="label switch")
        ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.8, label=f"chance={chance:.1f}%")
        ax.set_xlabel("Time relative to label switch (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Group Average M{task_m} Label-Switch Decoding (win={wl}s, n={len(group_results)})")
        ax.set_ylim([-5, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig_path = save_dir / f"GroupAverage_M{task_m}_win{int(wl * 1000)}ms.png"
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"Saved group average figure: {fig_path}")


# ═══════════════════════════════════════════════════
#  Main entry
# ═══════════════════════════════════════════════════

def process_subject(subj_dir: Path, save_root: Path):
    subject = subj_dir.name
    print(f"\n{'#' * 80}\nProcessing subject: {subject}\n{'#' * 80}")

    pairs = find_file_pairs(subj_dir)
    if not pairs:
        print(f"  [SKIP] {subject}: 未找到成对的 VEP_Exp2_M*_EEG/States_*.mat 文件。")
        return {}

    print(f"  Found {len(pairs)} EEG+States file pairs:")
    for p in pairs:
        print(f"    M{p.task_m} {p.timestamp}")

    raw_blocks_by_m: Dict[int, List[RawBlock]] = {}
    block_counters_by_m: Dict[int, int] = {}

    for pair in pairs:
        block_id = block_counters_by_m.get(pair.task_m, 0)
        try:
            block = load_file_pair(pair, block_id=block_id)
        except Exception as exc:
            print(f"  [WARN] 加载失败 {pair.eeg_path.name} / {pair.state_path.name}: {exc}")
            continue
        raw_blocks_by_m.setdefault(pair.task_m, []).append(block)
        block_counters_by_m[pair.task_m] = block_id + 1

    subject_results_by_m = {}
    for task_m, blocks in sorted(raw_blocks_by_m.items()):
        subj_m_dir = save_root / subject / f"M{task_m}"
        for step_size in STEP_SIZES:
            step_ms = int(round(step_size * 1000))
            step_dir = subj_m_dir / f"step{step_ms}ms"
            result, _ = process_task_blocks(
                subject,
                task_m,
                blocks,
                step_dir,
                METHODS,
                WIN_LENS,
                step_size,
                T_BEFORE,
                T_AFTER,
            )
            if result is not None:
                # If multiple step sizes are used, keep separate keys.
                subject_results_by_m.setdefault(task_m, {})[step_ms] = result

    return subject_results_by_m


def main():
    root = Path(ROOT_DIR)
    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)

    subj_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub")]) if root.exists() else []
    if not subj_dirs:
        print(f"在 {ROOT_DIR} 下未找到 sub* 文件夹。")
        return

    # group_results_by_m_step[M][step_ms][subject] = result
    group_results_by_m_step: Dict[int, Dict[int, Dict[str, dict]]] = {}

    for subj_dir in subj_dirs:
        subj_results = process_subject(subj_dir, save_root)
        for task_m, by_step in subj_results.items():
            for step_ms, result in by_step.items():
                group_results_by_m_step.setdefault(task_m, {}).setdefault(step_ms, {})[subj_dir.name] = result

    # Group average across subjects for each M and step.
    for task_m, by_step in sorted(group_results_by_m_step.items()):
        for step_ms, subj_res in sorted(by_step.items()):
            group_dir = save_root / f"GroupAverage_M{task_m}" / f"step{step_ms}ms"
            group_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(group_dir / f"GroupAverage_M{task_m}_step{step_ms}ms_subject_results.npy"), subj_res)
            plot_group_average(subj_res, task_m, group_dir)

    print("\nAll subjects processed.")


if __name__ == "__main__":
    main()
