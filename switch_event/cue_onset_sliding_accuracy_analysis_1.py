#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cue-onset sliding-window decoding accuracy for the continuous multi-class
same-frequency phase-coded SSVEP paradigm.

Compared with label_switch_accuracy_analysis.py, this script aligns the test
axis to each cue onset and scans windows from LATENCY to END_TIME. Training
uses the same phase-aligned augmentation idea, but excludes the complete cue
epoch currently being tested.

Default data layout:

Dataset/
  sub1/
    VEP_Exp2_M2_Active_EEG_20260714_201635.mat
    VEP_Exp2_M2_Active_States_20260714_201635.mat
    VEP_Exp2_M2_Passive_EEG_....mat
    VEP_Exp2_M2_Passive_States_....mat
    VEP_Exp2_M4_Active_EEG_....mat
    VEP_Exp2_M4_Active_States_....mat

Passive mode exists only for M=2.

Run from this folder or anywhere:

    python cue_onset_sliding_accuracy_analysis.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib-cache")))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from label_switch_accuracy_analysis import (
    FB_BANDS,
    FILTER_MARGIN_S,
    FS_DECIMATION,
    INTENSITY_THRESHOLD,
    USE_REREF,
    RawBlock,
    build_peaks_for_block,
    find_col_idx,
    find_file_pairs,
    find_nearest_peak,
    has_all_classes,
    load_file_pair,
    preprocess_block,
    train_and_predict,
)


# ═══════════════════════════════════════════════════
#  User-adjustable parameters
# ═══════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR / "Dataset"
SAVE_ROOT = SCRIPT_DIR / "Results_CueOnset_Multiclass"

# Window start times are scanned in [LATENCY, END_TIME - win_len].
LATENCY = 1.0
END_TIME = 2.0
WIN_LENS = [0.05, 0.10]
STEP_SIZE = 0.025

# Default split: for a test cue, exclude the full current cue epoch from
# training, where the cue epoch is onset -> next true_label change.
SPLIT_MODE = "exclude_current_cue_epoch"

# Phase-aligned augmentation reference. In global_reference mode this defines
# the common phase origin; labels still come from true_label.
PEAK_ALIGNMENT_MODE = "global_reference"
REFERENCE_INTENSITY_LABEL = 1

METHODS = {
    "Aug_LDA": True,
    "Aug_FB_LDA": True,
    "Aug_TRCA": True,
    "Aug_FB_TRCA": True,
    "Aug_eTRCA": True,
    "Aug_FB_eTRCA": True,
}

REQUIRE_ALL_CLASSES = True
PRINT_EVERY_N_OFFSETS = 5


# ═══════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class CueEvent:
    block: int
    cue_index: int
    onset: int
    end: int
    label_1b: int


# ═══════════════════════════════════════════════════
#  Cue and training-window construction
# ═══════════════════════════════════════════════════

def build_cue_events_for_block(
    block_id: int,
    true_label_ds: np.ndarray,
    task_m: int,
) -> List[CueEvent]:
    """
    Build contiguous cue epochs from true_label.

    A cue epoch starts when true_label enters 1..M or changes between valid
    labels. It ends at the next label change, invalid label, or file end.
    """
    labels = np.asarray(true_label_ds).astype(int)
    events: List[CueEvent] = []
    i = 0
    cue_index = 0
    while i < len(labels):
        lab = int(labels[i])
        if not (1 <= lab <= task_m):
            i += 1
            continue
        onset = i
        j = i + 1
        while j < len(labels) and int(labels[j]) == lab:
            j += 1
        events.append(
            CueEvent(
                block=block_id,
                cue_index=cue_index,
                onset=onset,
                end=j,
                label_1b=lab,
            )
        )
        cue_index += 1
        i = j
    return events


def generate_sliding_starts(latency: float, end_time: float, win_len: float, step_size: float) -> np.ndarray:
    """Return window start offsets in seconds, stopping before start+win_len exceeds END_TIME."""
    last_start = end_time - win_len
    if last_start + 1e-12 < latency:
        return np.array([], dtype=float)
    return np.arange(latency, last_start + 1e-12, step_size, dtype=float)


def _overlaps_any(start: int, end: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start < r_end and r_start < end for r_start, r_end in ranges)


def prepare_train_data_excluding_ranges(
    eeg_blocks: Dict[int, np.ndarray],
    peaks_by_block: Dict[int, dict],
    win_len_samples: int,
    tau1: int,
    fs: float,
    margin_s: float,
    exclude_ranges_by_block: Dict[int, List[Tuple[int, int]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training samples from reference peaks while excluding the current
    cue epoch. Labels are true_label values stored in the peak table.
    """
    margin_samples = int(round(margin_s * fs))
    X_list = []
    Y_list = []

    for block_id, eeg in eeg_blocks.items():
        T = eeg.shape[0]
        peaks = peaks_by_block[block_id]
        ranges = exclude_ranges_by_block.get(block_id, [])

        for peak_pos, peak_label in zip(peaks["positions"], peaks["labels"]):
            train_start = int(peak_pos) - int(tau1)
            train_end = train_start + win_len_samples
            if train_start < margin_samples or train_end > T:
                continue
            if _overlaps_any(train_start, train_end, ranges):
                continue
            seg = eeg[train_start:train_end, :]
            if seg.shape[0] != win_len_samples:
                continue
            X_list.append(seg.T)
            Y_list.append(int(peak_label) - 1)

    if not X_list:
        return np.empty((0,)), np.empty((0,), dtype=int)
    return np.stack(X_list), np.asarray(Y_list, dtype=int)


def _exclude_ranges_for_event(event: CueEvent) -> Dict[int, List[Tuple[int, int]]]:
    if SPLIT_MODE != "exclude_current_cue_epoch":
        raise ValueError(f"Unsupported SPLIT_MODE: {SPLIT_MODE}")
    return {int(event.block): [(int(event.onset), int(event.end))]}


# ═══════════════════════════════════════════════════
#  Analysis core
# ═══════════════════════════════════════════════════

def process_task_blocks(
    subject: str,
    task_m: int,
    paradigm_mode: str,
    raw_blocks: List[RawBlock],
    save_dir: Path,
):
    paradigm_mode = str(paradigm_mode).lower()
    if paradigm_mode not in {"active", "passive"}:
        raise ValueError(f"Unknown paradigm_mode: {paradigm_mode}")
    if paradigm_mode == "passive" and task_m != 2:
        raise ValueError("Passive mode is valid only for M=2.")
    if any(b.paradigm_mode != paradigm_mode for b in raw_blocks):
        raise ValueError(f"{subject} M{task_m}: raw_blocks contain mixed paradigm modes.")

    save_dir.mkdir(parents=True, exist_ok=True)
    fs_list = [b.fs_raw / FS_DECIMATION for b in raw_blocks]
    if max(fs_list) - min(fs_list) > 1e-6:
        raise ValueError(f"{subject} M{task_m}: multiple decimated Fs values: {fs_list}")
    fs = fs_list[0]

    print(f"\n{'=' * 72}")
    print(
        f"Subject={subject} | M={task_m} | mode={paradigm_mode.capitalize()} | "
        f"blocks={len(raw_blocks)} | Fs_ds={fs:.1f} Hz"
    )
    print(
        f"Cue scan: LATENCY={LATENCY:.3f}s, END_TIME={END_TIME:.3f}s, "
        f"step={STEP_SIZE:.3f}s | split={SPLIT_MODE}"
    )
    print(f"{'=' * 72}")

    eeg_wide_blocks: Dict[int, np.ndarray] = {}
    eeg_fb_blocks: Dict[int, List[np.ndarray]] = {}
    peaks_by_block: Dict[int, dict] = {}
    cue_events: List[CueEvent] = []

    for block in raw_blocks:
        true_label_col = find_col_idx(block.state_columns, "true_label")
        for k in range(1, task_m + 1):
            find_col_idx(block.state_columns, f"intensity_{k}")

        eeg_w, eeg_fb, states_ds, _ = preprocess_block(block, FS_DECIMATION, USE_REREF)
        eeg_wide_blocks[block.block_id] = eeg_w
        eeg_fb_blocks[block.block_id] = eeg_fb

        true_label_ds = states_ds[:, true_label_col]
        block_events = build_cue_events_for_block(block.block_id, true_label_ds, task_m)
        cue_events.extend(block_events)

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

        cue_counts = np.bincount([e.label_1b for e in block_events], minlength=task_m + 1)[1:]
        peak_counts = [len(peaks["positions_by_intensity"][k]) for k in range(1, task_m + 1)]
        train_peak_counts = np.bincount(peaks["labels"], minlength=task_m + 1)[1:]
        print(
            f"  block {block.block_id:02d} | {block.timestamp} | EEG_ds={eeg_w.shape} | "
            f"cues={len(block_events)} {cue_counts.tolist()} | "
            f"peaks/intensity={peak_counts} | selected_train_peaks={train_peak_counts.tolist()}"
        )

    if not cue_events:
        print(f"  [SKIP] {subject} M{task_m}: no valid cue events.")
        return None

    active_methods = [k for k, v in METHODS.items() if v]
    all_results = {}

    for win_len in WIN_LENS:
        win_samples = int(round(win_len * fs))
        slide_starts_s = generate_sliding_starts(LATENCY, END_TIME, win_len, STEP_SIZE)
        slide_starts_smp = np.round(slide_starts_s * fs).astype(int)
        n_offsets = len(slide_starts_s)

        print(
            f"\n  Window={win_len:.3f}s ({win_samples} samples) | "
            f"offsets={n_offsets} | cues={len(cue_events)}"
        )
        if n_offsets == 0:
            print("    [SKIP] win_len is longer than END_TIME-LATENCY.")
            continue

        acc_curves = {m: np.full(n_offsets, np.nan, dtype=float) for m in active_methods}
        correct_counts = {m: np.zeros(n_offsets, dtype=int) for m in active_methods}
        valid_counts = np.zeros(n_offsets, dtype=int)

        per_label_acc_curves = {
            m: np.full((task_m, n_offsets), np.nan, dtype=float) for m in active_methods
        }
        per_label_correct = {
            m: np.zeros((task_m, n_offsets), dtype=int) for m in active_methods
        }
        per_label_valid = np.zeros((task_m, n_offsets), dtype=int)

        train_size_records: List[int] = []
        skip_summary = {
            "test_boundary": 0,
            "test_crosses_cue_epoch": 0,
            "missing_reference_peak": 0,
            "empty_or_incomplete_train": 0,
            "fb_train_failed": 0,
        }

        for t_idx, (offset_s, offset_smp) in enumerate(zip(slide_starts_s, slide_starts_smp)):
            offset_correct = {m: 0 for m in active_methods}
            offset_valid = 0
            offset_label_correct = {m: np.zeros(task_m, dtype=int) for m in active_methods}
            offset_label_valid = np.zeros(task_m, dtype=int)

            for event in cue_events:
                blk = int(event.block)
                eeg_w = eeg_wide_blocks[blk]
                test_start = int(event.onset) + int(offset_smp)
                test_end = test_start + win_samples

                if test_start < 0 or test_end > eeg_w.shape[0]:
                    skip_summary["test_boundary"] += 1
                    continue
                if test_end > int(event.end):
                    skip_summary["test_crosses_cue_epoch"] += 1
                    continue

                true_label_0b = int(event.label_1b) - 1
                test_center = (test_start + test_end) // 2
                ref_peaks = peaks_by_block[blk]["positions_by_intensity"].get(
                    int(REFERENCE_INTENSITY_LABEL),
                    np.array([], dtype=int),
                )
                peak_pos, _ = find_nearest_peak(ref_peaks, test_center)
                if peak_pos is None:
                    skip_summary["missing_reference_peak"] += 1
                    continue
                tau1 = int(peak_pos) - int(test_start)

                exclude_ranges = _exclude_ranges_for_event(event)
                X_train, Y_train = prepare_train_data_excluding_ranges(
                    eeg_wide_blocks,
                    peaks_by_block,
                    win_samples,
                    tau1,
                    fs,
                    FILTER_MARGIN_S,
                    exclude_ranges,
                )
                if X_train.ndim != 3 or X_train.shape[0] == 0:
                    skip_summary["empty_or_incomplete_train"] += 1
                    continue
                if REQUIRE_ALL_CLASSES and not has_all_classes(Y_train, task_m):
                    skip_summary["empty_or_incomplete_train"] += 1
                    continue

                X_train_fb = []
                fb_ok = True
                for b_idx in range(len(FB_BANDS)):
                    fb_blocks_b = {bid: eeg_fb_blocks[bid][b_idx] for bid in eeg_fb_blocks}
                    Xb, _ = prepare_train_data_excluding_ranges(
                        fb_blocks_b,
                        peaks_by_block,
                        win_samples,
                        tau1,
                        fs,
                        FILTER_MARGIN_S,
                        exclude_ranges,
                    )
                    if Xb.ndim != 3 or Xb.shape[0] != X_train.shape[0]:
                        fb_ok = False
                    X_train_fb.append(Xb)
                if not fb_ok:
                    skip_summary["fb_train_failed"] += 1
                    continue

                test_sample = eeg_w[test_start:test_end, :].T
                test_sample_fb = np.stack(
                    [
                        eeg_fb_blocks[blk][b][test_start:test_end, :].T
                        for b in range(len(FB_BANDS))
                    ],
                    axis=0,
                )

                preds = train_and_predict(
                    X_train,
                    Y_train,
                    X_train_fb,
                    test_sample,
                    test_sample_fb,
                    METHODS,
                    task_m,
                )

                train_size_records.append(int(X_train.shape[0]))
                offset_valid += 1
                offset_label_valid[true_label_0b] += 1
                for m_name, pred in preds.items():
                    if pred == true_label_0b:
                        offset_correct[m_name] += 1
                        offset_label_correct[m_name][true_label_0b] += 1

            valid_counts[t_idx] = offset_valid
            per_label_valid[:, t_idx] = offset_label_valid
            for m_name in active_methods:
                correct_counts[m_name][t_idx] = offset_correct[m_name]
                if offset_valid > 0:
                    acc_curves[m_name][t_idx] = offset_correct[m_name] / offset_valid
                per_label_correct[m_name][:, t_idx] = offset_label_correct[m_name]
                for lab_idx in range(task_m):
                    if offset_label_valid[lab_idx] > 0:
                        per_label_acc_curves[m_name][lab_idx, t_idx] = (
                            offset_label_correct[m_name][lab_idx] / offset_label_valid[lab_idx]
                        )

            if t_idx % PRINT_EVERY_N_OFFSETS == 0 or t_idx == n_offsets - 1:
                acc_txt = " ".join(
                    [
                        f"{m}:{acc_curves[m][t_idx] * 100:5.1f}%"
                        if np.isfinite(acc_curves[m][t_idx]) else f"{m}:  nan"
                        for m in active_methods
                    ]
                )
                print(f"    t={offset_s:.3f}s | valid={offset_valid}/{len(cue_events)} | {acc_txt}")

        full_mean_acc = {m: float(np.nanmean(acc_curves[m])) for m in active_methods}
        per_label_full_mean_acc = {
            m: np.nanmean(per_label_acc_curves[m], axis=1) for m in active_methods
        }

        result = {
            "slide_starts_s": slide_starts_s,
            "win_len_s": win_len,
            "step_size_s": STEP_SIZE,
            "latency_s": LATENCY,
            "end_time_s": END_TIME,
            "task_m": task_m,
            "paradigm_mode": paradigm_mode,
            "chance_level": 1.0 / task_m,
            "acc_curves": acc_curves,
            "correct_counts": correct_counts,
            "valid_counts": valid_counts,
            "per_label_acc_curves": per_label_acc_curves,
            "per_label_correct": per_label_correct,
            "per_label_valid": per_label_valid,
            "full_mean_acc": full_mean_acc,
            "per_label_full_mean_acc": per_label_full_mean_acc,
            "train_size_stats": _summarize_train_sizes(train_size_records),
            "skip_summary": skip_summary,
            "split_mode": SPLIT_MODE,
            "peak_alignment_mode": PEAK_ALIGNMENT_MODE,
            "reference_intensity_label": REFERENCE_INTENSITY_LABEL,
        }
        all_results[win_len] = result

        mean_txt = " ".join([f"{m}:{full_mean_acc[m] * 100:5.1f}%" for m in active_methods])
        print(f"    Full-window mean accuracy | {mean_txt}")
        print(f"    Skip summary: {skip_summary}")

    if not all_results:
        return None

    _save_task_outputs(
        subject, paradigm_mode, task_m, raw_blocks, cue_events, all_results, save_dir
    )
    return all_results


def _summarize_train_sizes(train_sizes: List[int]) -> Optional[dict]:
    if not train_sizes:
        return None
    arr = np.asarray(train_sizes)
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "count": int(arr.size),
    }


# ═══════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════

def _save_task_outputs(
    subject: str,
    paradigm_mode: str,
    task_m: int,
    raw_blocks: List[RawBlock],
    cue_events: List[CueEvent],
    all_results: dict,
    save_dir: Path,
):
    win_tag = "-".join(f"{int(w * 1000)}ms" for w in sorted(all_results))
    mode_name = paradigm_mode.capitalize()
    out_npy = save_dir / f"CueOnset_{subject}_M{task_m}_{mode_name}_wins{win_tag}.npy"
    np.save(str(out_npy), all_results)
    print(f"  Saved results: {out_npy}")

    stats_path = (
        save_dir / f"CueOnset_{subject}_M{task_m}_{mode_name}_wins{win_tag}_stats.txt"
    )
    _save_stats_text(
        stats_path, subject, paradigm_mode, task_m, raw_blocks, cue_events, all_results
    )
    _plot_task_results(all_results, subject, task_m, paradigm_mode, save_dir)


def _save_stats_text(
    path: Path,
    subject: str,
    paradigm_mode: str,
    task_m: int,
    raw_blocks: List[RawBlock],
    cue_events: List[CueEvent],
    all_results: dict,
):
    cue_counts = np.bincount([e.label_1b for e in cue_events], minlength=task_m + 1)[1:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Subject: {subject}\n")
        f.write(f"Task M: {task_m}\n")
        f.write(f"Paradigm mode: {paradigm_mode}\n")
        f.write(f"Blocks: {len(raw_blocks)}\n")
        f.write(f"Cue events: {len(cue_events)}\n")
        f.write(f"Cue counts by label: {cue_counts.tolist()}\n")
        f.write(f"Chance level: {100 / task_m:.2f}%\n")
        f.write(f"LATENCY: {LATENCY:.3f}s\n")
        f.write(f"END_TIME: {END_TIME:.3f}s\n")
        f.write(f"STEP_SIZE: {STEP_SIZE:.3f}s\n")
        f.write(f"WIN_LENS: {WIN_LENS}\n")
        f.write(f"SPLIT_MODE: {SPLIT_MODE}\n")
        f.write(f"PEAK_ALIGNMENT_MODE: {PEAK_ALIGNMENT_MODE}\n")
        f.write(f"REFERENCE_INTENSITY_LABEL: {REFERENCE_INTENSITY_LABEL}\n")
        f.write(f"INTENSITY_THRESHOLD: {INTENSITY_THRESHOLD}\n")
        f.write("\nBlock files:\n")
        for b in raw_blocks:
            f.write(f"  block {b.block_id}: {b.timestamp}, Fs={b.fs_raw}, EEG={b.eeg.shape}, States={b.states.shape}\n")

        for win_len, res in sorted(all_results.items()):
            f.write(f"\nWindow {win_len:.3f}s\n")
            valid = res["valid_counts"]
            f.write(f"  valid_tests: min={np.nanmin(valid)}, max={np.nanmax(valid)}, mean={np.nanmean(valid):.1f}\n")
            f.write(f"  train_size_stats: {res['train_size_stats']}\n")
            f.write(f"  skip_summary: {res['skip_summary']}\n")
            for m_name, mean_acc in res["full_mean_acc"].items():
                per_label = res["per_label_full_mean_acc"][m_name]
                f.write(
                    f"  {m_name}: full_mean={mean_acc * 100:.2f}%, "
                    f"per_label={np.round(per_label * 100, 2).tolist()}\n"
                )
    print(f"  Saved stats: {path}")


def _method_colors() -> Dict[str, str]:
    return {
        "Aug_LDA": "#1f77b4",
        "Aug_FB_LDA": "#ff7f0e",
        "Aug_TRCA": "#2ca02c",
        "Aug_FB_TRCA": "#d62728",
        "Aug_eTRCA": "#9467bd",
        "Aug_FB_eTRCA": "#8c564b",
    }


def _plot_task_results(
    all_results: dict, subject: str, task_m: int, paradigm_mode: str, save_dir: Path
):
    colors = _method_colors()
    chance = 100.0 / task_m
    labels = [f"label {i}" for i in range(1, task_m + 1)]

    for win_len, res in sorted(all_results.items()):
        t_ms = res["slide_starts_s"] * 1000.0

        fig, ax = plt.subplots(figsize=(12, 6))
        for m_name, acc in res["acc_curves"].items():
            style = "-" if m_name == "Aug_FB_LDA" else "--"
            ax.plot(
                t_ms,
                acc * 100.0,
                color=colors.get(m_name, "gray"),
                linestyle=style,
                linewidth=2.0,
                label=f"{m_name} mean={res['full_mean_acc'][m_name] * 100:.1f}%",
            )
        ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.8, label=f"chance={chance:.1f}%")
        ax.set_xlabel("Time from cue onset (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(
            f"{subject} M{task_m} {paradigm_mode.capitalize()} "
            f"Cue-Onset Accuracy (win={win_len:.3f}s)"
        )
        ax.set_ylim([-5, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig_path = save_dir / (
            f"CueOnset_{subject}_M{task_m}_{paradigm_mode.capitalize()}_"
            f"win{int(win_len * 1000)}ms_overall.png"
        )
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")

        methods = list(res["acc_curves"].keys())
        n_cols = 3
        n_rows = int(np.ceil(len(methods) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.asarray(axes).reshape(-1)
        label_colors = plt.cm.tab10(np.linspace(0, 1, max(task_m, 2)))
        for idx, m_name in enumerate(methods):
            ax = axes[idx]
            per_label = res["per_label_acc_curves"][m_name]
            for lab_idx in range(task_m):
                ax.plot(
                    t_ms,
                    per_label[lab_idx] * 100.0,
                    color=label_colors[lab_idx],
                    linewidth=1.6,
                    label=labels[lab_idx],
                )
            ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.8)
            ax.set_title(m_name)
            ax.set_ylim([-5, 105])
            ax.set_xlabel("Time from cue onset (ms)")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes[len(methods):]:
            ax.axis("off")
        fig.suptitle(
            f"{subject} M{task_m} {paradigm_mode.capitalize()} "
            f"Per-Label Accuracy (win={win_len:.3f}s)",
            fontsize=14,
        )
        fig.tight_layout()
        fig_path = save_dir / (
            f"CueOnset_{subject}_M{task_m}_{paradigm_mode.capitalize()}_"
            f"win{int(win_len * 1000)}ms_per_label.png"
        )
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"  Saved per-label figure: {fig_path}")


def plot_group_full_mean(
    group_results_by_condition: Dict[Tuple[int, str], Dict[str, dict]],
    save_root: Path,
):
    """
    Plot subject-level full-window means, keeping Active and Passive separate.

    Active may contain M2/M4/M9. Passive is accepted only for M2.
    """
    if not group_results_by_condition:
        return
    summary_dir = save_root / "GroupSummary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    modes = sorted({mode for (_, mode) in group_results_by_condition})
    colors = _method_colors()

    for paradigm_mode in modes:
        task_ms = sorted(
            m for (m, mode) in group_results_by_condition if mode == paradigm_mode
        )
        if not task_ms:
            continue
        if paradigm_mode == "passive" and any(m != 2 for m in task_ms):
            raise ValueError("Passive group results must contain only M=2.")

        first_m = task_ms[0]
        first_subject = next(
            iter(group_results_by_condition[(first_m, paradigm_mode)])
        )
        sample = group_results_by_condition[(first_m, paradigm_mode)][first_subject]
        win_lens = sorted(sample)
        methods = list(sample[win_lens[0]]["acc_curves"].keys())
        mode_name = paradigm_mode.capitalize()

        for win_len in win_lens:
            fig, ax = plt.subplots(figsize=(9, 6))
            for m_name in methods:
                means = []
                sems = []
                for task_m in task_ms:
                    subj_vals = []
                    for subj_res in group_results_by_condition.get(
                        (task_m, paradigm_mode), {}
                    ).values():
                        if win_len in subj_res:
                            subj_vals.append(
                                subj_res[win_len]["full_mean_acc"][m_name]
                            )
                    if subj_vals:
                        arr = np.asarray(subj_vals, dtype=float)
                        means.append(float(np.nanmean(arr)))
                        sems.append(
                            float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))
                            if arr.size > 1
                            else 0.0
                        )
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)
                ax.errorbar(
                    task_ms,
                    np.asarray(means) * 100.0,
                    yerr=np.asarray(sems) * 100.0,
                    marker="o",
                    linewidth=2.0,
                    capsize=4,
                    color=colors.get(m_name, None),
                    label=m_name,
                )
            ax.set_xlabel("Number of classes (M)")
            ax.set_ylabel("Full-window mean accuracy (%)")
            ax.set_title(
                f"Group Full-Window Mean Accuracy - {mode_name} "
                f"(win={win_len:.3f}s)"
            )
            ax.set_xticks(task_ms)
            ax.set_ylim([-5, 105])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig_path = summary_dir / (
                f"GroupFullMean_{mode_name}_win{int(win_len * 1000)}ms.png"
            )
            fig.savefig(str(fig_path), dpi=150)
            plt.close(fig)
            print(f"Saved group summary figure: {fig_path}")


# ═══════════════════════════════════════════════════
#  Main entry and smoke test
# ═══════════════════════════════════════════════════

def process_subject(subj_dir: Path, save_root: Path):
    subject = subj_dir.name
    print(f"\n{'#' * 80}\nProcessing subject: {subject}\n{'#' * 80}")

    pairs = find_file_pairs(subj_dir)
    if not pairs:
        print(
            f"  [SKIP] {subject}: no paired "
            "VEP_Exp2_M*_{Active|Passive}_EEG/States_*.mat files."
        )
        return {}

    raw_blocks_by_condition: Dict[Tuple[int, str], List[RawBlock]] = {}
    block_counters_by_condition: Dict[Tuple[int, str], int] = {}
    for pair in pairs:
        condition = (pair.task_m, pair.paradigm_mode)
        block_id = block_counters_by_condition.get(condition, 0)
        try:
            block = load_file_pair(pair, block_id=block_id)
        except Exception as exc:
            print(f"  [WARN] failed to load {pair.eeg_path.name} / {pair.state_path.name}: {exc}")
            continue
        raw_blocks_by_condition.setdefault(condition, []).append(block)
        block_counters_by_condition[condition] = block_id + 1

    subject_results_by_condition = {}
    for (task_m, paradigm_mode), blocks in sorted(raw_blocks_by_condition.items()):
        if paradigm_mode == "passive" and task_m != 2:
            print(
                f"  [WARN] {subject}: M{task_m} Passive is invalid; "
                "Passive is supported only for M=2. Skipping."
            )
            continue
        mode_name = paradigm_mode.capitalize()
        out_dir = save_root / subject / f"M{task_m}_{mode_name}"
        result = process_task_blocks(
            subject, task_m, paradigm_mode, blocks, out_dir
        )
        if result is not None:
            subject_results_by_condition[(task_m, paradigm_mode)] = result
    return subject_results_by_condition


def main():
    import sys

    if "--smoke-test" in sys.argv:
        _run_smoke_test()
        return

    save_root = Path(SAVE_ROOT)
    save_root.mkdir(parents=True, exist_ok=True)
    root = Path(ROOT_DIR)
    subj_dirs = (
        sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("sub")])
        if root.exists()
        else []
    )
    if not subj_dirs:
        print(f"No sub* folders found under {root}")
        return

    # group_results_by_condition[(M, mode)][subject] = result
    group_results_by_condition: Dict[Tuple[int, str], Dict[str, dict]] = {}
    for subj_dir in subj_dirs:
        subj_results = process_subject(subj_dir, save_root)
        for condition, result in subj_results.items():
            group_results_by_condition.setdefault(condition, {})[subj_dir.name] = result

    plot_group_full_mean(group_results_by_condition, save_root)


def _run_smoke_test():
    labels = np.array([0, 1, 1, 2, 2, 2, 1, 1, 0, 2, 2])
    events = build_cue_events_for_block(0, labels, task_m=2)
    assert [(e.onset, e.end, e.label_1b) for e in events] == [(1, 3, 1), (3, 6, 2), (6, 8, 1), (9, 11, 2)]

    starts = generate_sliding_starts(0.0, 0.20, 0.05, 0.025)
    assert np.allclose(starts, [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15])

    eeg_blocks = {0: np.arange(200 * 2, dtype=float).reshape(200, 2)}
    peaks_by_block = {
        0: {
            "positions": np.array([20, 40, 60, 80, 100, 120]),
            "labels": np.array([1, 2, 1, 2, 1, 2]),
        }
    }
    X, Y = prepare_train_data_excluding_ranges(
        eeg_blocks=eeg_blocks,
        peaks_by_block=peaks_by_block,
        win_len_samples=10,
        tau1=3,
        fs=100.0,
        margin_s=0.0,
        exclude_ranges_by_block={0: [(55, 85)]},
    )
    assert X.shape == (4, 2, 10)
    assert Y.tolist() == [0, 1, 0, 1]
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
