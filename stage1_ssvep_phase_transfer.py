#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 single-stimulus SSVEP phase-transfer experiment.

This script presents one central flickering square per trial. The stimulus
frequency is fixed, and each trial starts with one configured initial phase.
It is intended to collect raw EEG, triggers, and trial timing metadata for
later phase-transfer analysis.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_TRUE,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnd,
    glFlush,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glVertex2f,
    glViewport,
)
from scipy.io import savemat


try:
    from eConEXG import iRecorder
    import serial

    HARDWARE_LIBS_AVAILABLE = True
except ImportError:
    HARDWARE_LIBS_AVAILABLE = False
    print("[Stage1] Hardware libraries not found. Online mode will be disabled.")


@dataclass
class Stage1Config:
    online_mode: bool = True
    fs: int = 2000
    port: str = "COM7"
    required_refresh_rate: Optional[int] = 600
    random_seed: Optional[int] = None

    stim_freq: float = 10.0
    phase_count: int = 6
    phase_step_deg: Optional[float] = None
    initial_phase_deg: float = 0.0
    block_count: int = 1
    trials_per_phase_per_block: int = 10

    square_scale: float = 0.22
    stimulus_duration_sec: float = 2.5
    iti_min_sec: float = 0.8
    iti_max_sec: float = 1.2
    block_break_sec: float = 10.0

    trigger_base: int = 1
    save_prefix: str = "Stage1_SSVEP_PhaseTransfer"
    waveform: str = "square"
    debug_drop_print_limit: int = 5

    def phases_deg(self) -> list[float]:
        step = 360.0 / self.phase_count if self.phase_step_deg is None else self.phase_step_deg
        return [float((self.initial_phase_deg + step * i) % 360.0) for i in range(self.phase_count)]


class NiantongPort:
    def __init__(self, port_addr: str, baudrate: int = 115200):
        self.valid = False
        self.port = None
        try:
            if HARDWARE_LIBS_AVAILABLE:
                self.port = serial.Serial(port=port_addr, baudrate=baudrate)
                self.valid = True
        except Exception as exc:
            print(f"[Stage1] Serial port error: {exc}", flush=True)

    def set_data(self, label: int) -> None:
        if not self.valid or self.port is None:
            return
        if label < 1 or label > 255:
            print("[Stage1] Trigger label must be in range 1..255.", flush=True)
            return

        label_hex = format(label, "02X")
        frame_end = "55660D"
        try:
            self.port.write(bytes.fromhex(label_hex + frame_end))
        except Exception as exc:
            print(f"[Stage1] Serial write error: {exc}", flush=True)

    def stop(self) -> None:
        if self.valid and self.port is not None and self.port.is_open:
            self.port.close()


class DataAcquisitionThread(threading.Thread):
    def __init__(self, config: Stage1Config):
        super().__init__(daemon=True)
        self.config = config
        self.online_mode = config.online_mode and HARDWARE_LIBS_AVAILABLE
        self.running = True
        self.eeg_dev = None
        self.trigger = None
        self.eeg_data = np.empty((0, 9), dtype=np.float32)

    def init_devices(self) -> bool:
        if not self.online_mode:
            print("[Stage1] Offline mode: skipping EEG and trigger devices.", flush=True)
            return True

        try:
            self.eeg_dev = iRecorder(dev_type="USB8")
            self.eeg_dev.set_frequency(self.config.fs)
            self.eeg_dev.find_devs()
            while self.running:
                devices = self.eeg_dev.get_devs()
                if devices:
                    break
                time.sleep(0.1)
            if not self.running:
                return False

            self.eeg_dev.connect_device(devices[0])
            self.eeg_dev.start_acquisition_data(with_q=True)
            self.trigger = NiantongPort(self.config.port)
            return True
        except Exception as exc:
            print(f"[Stage1] Device initialization error: {exc}", flush=True)
            return False

    def run(self) -> None:
        if not self.init_devices():
            return

        while self.running:
            if self.online_mode:
                try:
                    frames = self.eeg_dev.get_data(timeout=0.01)
                    if frames:
                        frame_array = np.asarray(frames, dtype=np.float32)
                        self.eeg_data = np.concatenate((self.eeg_data, frame_array), axis=0)
                except Exception as exc:
                    print(f"[Stage1] Data acquisition error: {exc}", flush=True)
                    break
            else:
                time.sleep(0.01)

    def send_trigger(self, value: int) -> None:
        if self.online_mode and self.trigger is not None:
            self.trigger.set_data(value)

    def stop(self, metadata: dict[str, object]) -> Optional[str]:
        self.running = False
        filename = None

        if self.online_mode:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.save_prefix}_{timestamp}.mat"
            try:
                save_dict = {"data": self.eeg_data, "Fs": self.config.fs}
                save_dict.update(metadata)
                savemat(filename, save_dict)
                print(f"[Stage1] Data saved to {filename}", flush=True)
            except Exception as exc:
                print(f"[Stage1] Data saving error: {exc}", flush=True)

            if self.eeg_dev is not None:
                try:
                    self.eeg_dev.stop_acquisition()
                    self.eeg_dev.close_dev()
                except Exception:
                    pass
            if self.trigger is not None:
                self.trigger.stop()

        if threading.current_thread() is not self:
            self.join(timeout=5.0)
        return filename


class Stage1PhaseTransferExperiment:
    def __init__(self, config: Stage1Config):
        self.config = config
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.window = None
        self.width = 0
        self.height = 0
        self.refresh_rate = 0
        self.period_frames = 0.0
        self.trials: list[dict[str, float | int]] = []
        self.trial_log: list[tuple[int, int, float, int]] = []
        self.is_closing = False
        self.saved_filename: Optional[str] = None

        self.data_thread = DataAcquisitionThread(config)
        self._init_ui()
        self._validate_timing()
        self.trials = self._make_trial_schedule()
        self.data_thread.start()

    def _init_ui(self) -> None:
        mode_str = "ONLINE" if self.config.online_mode else "OFFLINE"
        if not glfw.init():
            sys.exit("GLFW init failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.refresh_rate = int(mode.refresh_rate)
        self.width = int(mode.size.width)
        self.height = int(mode.size.height)
        self.square_side = min(self.width, self.height) * self.config.square_scale

        glfw.window_hint(glfw.DOUBLEBUFFER, GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.REFRESH_RATE, self.refresh_rate)

        self.window = glfw.create_window(
            self.width,
            self.height,
            f"Stage1 SSVEP Phase Transfer - {mode_str}",
            monitor,
            None,
        )
        if not self.window:
            glfw.terminate()
            sys.exit("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        glfw.set_key_callback(self.window, self._key_callback)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-self.width / 2, self.width / 2, -self.height / 2, self.height / 2, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _validate_timing(self) -> None:
        cfg = self.config
        if cfg.stim_freq <= 0:
            raise ValueError("stim_freq must be > 0.")
        if cfg.phase_count < 2:
            raise ValueError("phase_count must be >= 2.")
        if cfg.block_count < 1:
            raise ValueError("block_count must be >= 1.")
        if cfg.trials_per_phase_per_block < 1:
            raise ValueError("trials_per_phase_per_block must be >= 1.")
        if cfg.square_scale <= 0:
            raise ValueError("square_scale must be > 0.")
        if cfg.stimulus_duration_sec <= 0:
            raise ValueError("stimulus_duration_sec must be > 0.")
        if cfg.iti_min_sec < 0 or cfg.iti_max_sec < 0:
            raise ValueError("ITI values must be >= 0.")
        if cfg.block_break_sec < 0:
            raise ValueError("block_break_sec must be >= 0.")
        if cfg.iti_min_sec > cfg.iti_max_sec:
            raise ValueError("iti_min_sec must be <= iti_max_sec.")
        if cfg.waveform not in {"square", "sine"}:
            raise ValueError("waveform must be 'square' or 'sine'.")

        self.period_frames = self.refresh_rate / cfg.stim_freq
        if cfg.required_refresh_rate is not None and self.refresh_rate != cfg.required_refresh_rate:
            print(
                f"[Stage1] WARNING: expected {cfg.required_refresh_rate} Hz, "
                f"detected {self.refresh_rate} Hz.",
                flush=True,
            )

        rounded_period = round(self.period_frames)
        if not math.isclose(self.period_frames, rounded_period, abs_tol=1e-6):
            print(
                f"[Stage1] WARNING: refresh/stim_freq = {self.period_frames:.6f} frames; "
                "non-integer periods can make phase realization less exact.",
                flush=True,
            )

        phases = cfg.phases_deg()
        rounded_phases = {round(phase, 6) for phase in phases}
        if len(rounded_phases) != len(phases):
            print(
                "[Stage1] WARNING: duplicate phase values detected after wrapping to 0..360 deg.",
                flush=True,
            )

        phase_frame_offsets = [phase / 360.0 * self.period_frames for phase in phases]
        non_integer_offsets = [
            offset for offset in phase_frame_offsets if not math.isclose(offset, round(offset), abs_tol=1e-6)
        ]
        if non_integer_offsets:
            print(
                "[Stage1] WARNING: some phase offsets are not integer frame offsets. "
                "Use a photodiode check before formal recording.",
                flush=True,
            )

        max_trigger = cfg.trigger_base + cfg.phase_count - 1
        if cfg.trigger_base < 1 or max_trigger > 255:
            raise ValueError("Trigger range must stay within 1..255.")

        print(
            f"[Stage1] Display: {self.width}x{self.height} @ {self.refresh_rate} Hz; "
            f"stim={cfg.stim_freq:.3f} Hz; period={self.period_frames:.3f} frames.",
            flush=True,
        )
        print(
            f"[Stage1] M={cfg.phase_count}; phases(deg)="
            f"{', '.join(f'{p:.1f}' for p in phases)}; "
            f"square={self.square_side:.1f}px; blocks={cfg.block_count}.",
            flush=True,
        )
        print(
            "[Stage1] Press SPACE only once at experiment start; "
            "block breaks continue automatically.",
            flush=True,
        )

    def _make_trial_schedule(self) -> list[dict[str, float | int]]:
        cfg = self.config
        phases_deg = cfg.phases_deg()
        phases_rad = [math.radians(phase) for phase in phases_deg]
        stim_frames = max(1, int(round(cfg.stimulus_duration_sec * self.refresh_rate)))
        block_break_frames = max(0, int(round(cfg.block_break_sec * self.refresh_rate)))
        frame_cursor = 0
        trials: list[dict[str, float | int]] = []
        trial_idx = 0

        for block_idx in range(cfg.block_count):
            labels = []
            for phase_idx in range(cfg.phase_count):
                labels.extend([phase_idx] * cfg.trials_per_phase_per_block)
            random.shuffle(labels)

            for trial_in_block, phase_idx in enumerate(labels):
                iti_sec = random.uniform(cfg.iti_min_sec, cfg.iti_max_sec)
                iti_frames = max(1, int(round(iti_sec * self.refresh_rate)))
                trigger = cfg.trigger_base + phase_idx
                onset_frame = frame_cursor
                stim_end_frame = onset_frame + stim_frames
                trial_end_frame = stim_end_frame + iti_frames
                trials.append(
                    {
                        "trial_idx": trial_idx,
                        "block_idx": block_idx,
                        "trial_in_block": trial_in_block,
                        "phase_idx": phase_idx,
                        "label": phase_idx + 1,
                        "phase_deg": phases_deg[phase_idx],
                        "phase_rad": phases_rad[phase_idx],
                        "trigger": trigger,
                        "onset_frame": onset_frame,
                        "stim_end_frame": stim_end_frame,
                        "trial_end_frame": trial_end_frame,
                        "stim_frames": stim_frames,
                        "iti_frames": iti_frames,
                    }
                )
                trial_idx += 1
                frame_cursor = trial_end_frame

            if block_idx < cfg.block_count - 1:
                frame_cursor += block_break_frames

        total_sec = frame_cursor / self.refresh_rate
        print(
            f"[Stage1] Trials={len(trials)} "
            f"({cfg.trials_per_phase_per_block * cfg.block_count} per phase total); "
            f"planned duration={total_sec / 60.0:.2f} min.",
            flush=True,
        )
        return trials

    def _key_callback(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def _draw_square(self, intensity: float) -> None:
        half = self.square_side / 2.0
        glColor3f(float(intensity), float(intensity), float(intensity))
        glBegin(GL_QUADS)
        glVertex2f(-half, -half)
        glVertex2f(half, -half)
        glVertex2f(half, half)
        glVertex2f(-half, half)
        glEnd()

    def _stimulus_intensity(self, local_frame: int, phase_rad: float) -> float:
        theta = 2.0 * math.pi * self.config.stim_freq * local_frame / self.refresh_rate + phase_rad
        value = math.sin(theta)
        if self.config.waveform == "sine":
            return 0.5 + 0.5 * value
        return 1.0 if value >= 0.0 else 0.0

    def _render_frame(self, frame_idx: int, trial: Optional[dict[str, float | int]]) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        intensity = 0.0
        if trial is not None:
            onset = int(trial["onset_frame"])
            stim_end = int(trial["stim_end_frame"])
            if onset <= frame_idx < stim_end:
                local_frame = frame_idx - onset
                intensity = self._stimulus_intensity(local_frame, float(trial["phase_rad"]))

        self._draw_square(intensity)
        glFlush()

    def _wait_for_space(self) -> bool:
        print("[Stage1] Press SPACE to start; ESC to quit.", flush=True)
        while not glfw.window_should_close(self.window):
            if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
                return True
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(0.01)
        return False

    def _schedule_matrix(self) -> np.ndarray:
        columns = [
            "trial_idx",
            "block_idx",
            "trial_in_block",
            "label",
            "phase_idx",
            "phase_deg",
            "phase_rad",
            "trigger",
            "onset_frame",
            "stim_end_frame",
            "trial_end_frame",
            "stim_frames",
            "iti_frames",
        ]
        return np.array([[trial[column] for column in columns] for trial in self.trials], dtype=np.float64)

    @staticmethod
    def _schedule_columns() -> np.ndarray:
        return np.array(
            [
                "trial_idx",
                "block_idx",
                "trial_in_block",
                "label",
                "phase_idx",
                "phase_deg",
                "phase_rad",
                "trigger",
                "onset_frame",
                "stim_end_frame",
                "trial_end_frame",
                "stim_frames",
                "iti_frames",
            ],
            dtype=object,
        )

    def _metadata(
        self,
        stimulus_time: float,
        measured_fps: float,
        dropped_frames: int,
        max_frame_dt: float,
    ) -> dict[str, object]:
        cfg = self.config
        trial_log = np.array(self.trial_log, dtype=np.float64) if self.trial_log else np.empty((0, 4))
        return {
            "experiment_name": np.array(["stage1_ssvep_phase_transfer"], dtype=object),
            "online_mode": int(cfg.online_mode),
            "refresh_rate": float(self.refresh_rate),
            "measured_fps": float(measured_fps),
            "stimulus_time_sec": float(stimulus_time),
            "stim_freq": float(cfg.stim_freq),
            "period_frames": float(self.period_frames),
            "phase_count": int(cfg.phase_count),
            "phases_deg": np.array(cfg.phases_deg(), dtype=np.float64),
            "phases_rad": np.radians(np.array(cfg.phases_deg(), dtype=np.float64)),
            "phase_step_deg": float(360.0 / cfg.phase_count if cfg.phase_step_deg is None else cfg.phase_step_deg),
            "square_size_px": float(self.square_side),
            "block_count": int(cfg.block_count),
            "trials_per_phase_per_block": int(cfg.trials_per_phase_per_block),
            "stimulus_duration_sec": float(cfg.stimulus_duration_sec),
            "iti_min_sec": float(cfg.iti_min_sec),
            "iti_max_sec": float(cfg.iti_max_sec),
            "block_break_sec": float(cfg.block_break_sec),
            "block_break_frames": int(round(cfg.block_break_sec * self.refresh_rate)),
            "waveform": np.array([cfg.waveform], dtype=object),
            "trigger_base": int(cfg.trigger_base),
            "trial_schedule": self._schedule_matrix(),
            "trial_schedule_columns": self._schedule_columns(),
            "trial_log": trial_log,
            "trial_log_columns": np.array(["trial_idx", "onset_frame", "onset_time_sec", "trigger"], dtype=object),
            "dropped_frames": int(dropped_frames),
            "max_frame_dt_sec": float(max_frame_dt),
        }

    def close_application(self, metadata: dict[str, object]) -> None:
        if self.is_closing:
            return
        self.is_closing = True
        print("[Stage1] Stopping experiment safely...", flush=True)
        if self.data_thread.is_alive():
            self.saved_filename = self.data_thread.stop(metadata)
        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
        except Exception:
            pass
        glfw.terminate()

    def run(self) -> None:
        if not self._wait_for_space():
            self.close_application({})
            return

        frame_idx = 0
        current_trial_idx = 0
        total_frames = int(self.trials[-1]["trial_end_frame"]) if self.trials else 0
        expected_dt = 1.0 / self.refresh_rate
        dropped_frames = 0
        max_frame_dt = 0.0
        t_start = time.perf_counter()
        t_prev = t_start

        try:
            while not glfw.window_should_close(self.window):
                if frame_idx >= total_frames:
                    break

                while (
                    current_trial_idx < len(self.trials)
                    and frame_idx >= int(self.trials[current_trial_idx]["trial_end_frame"])
                ):
                    current_trial_idx += 1

                trial = self.trials[current_trial_idx] if current_trial_idx < len(self.trials) else None
                if trial is not None and frame_idx == int(trial["onset_frame"]):
                    trigger = int(trial["trigger"])
                    self.data_thread.send_trigger(trigger)
                    self.trial_log.append(
                        (
                            int(trial["trial_idx"]),
                            int(trial["onset_frame"]),
                            time.perf_counter() - t_start,
                            trigger,
                        )
                    )

                self._render_frame(frame_idx, trial)
                glfw.swap_buffers(self.window)
                glfw.poll_events()

                t_now = time.perf_counter()
                dt = t_now - t_prev
                max_frame_dt = max(max_frame_dt, dt)
                if frame_idx > 5 and dt > expected_dt * 1.5:
                    dropped_frames += 1
                    if dropped_frames <= self.config.debug_drop_print_limit:
                        print(
                            f"[Stage1] Dropped-frame warning: frame={frame_idx}, "
                            f"dt={dt * 1000:.3f} ms, "
                            f"threshold={expected_dt * 1.5 * 1000:.3f} ms",
                            flush=True,
                        )
                t_prev = t_now
                frame_idx += 1
        finally:
            t_end = time.perf_counter()
            stimulus_time = t_end - t_start
            measured_fps = frame_idx / max(stimulus_time, 1e-9)
            drop_rate = dropped_frames / max(frame_idx, 1) * 100.0

            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(1.0)

            metadata = self._metadata(stimulus_time, measured_fps, dropped_frames, max_frame_dt)
            self.close_application(metadata)

            print("\n========== Stage1 SSVEP Phase Transfer Summary ==========")
            print(f"Stimulus time:       {stimulus_time:.2f} s")
            print(f"Total frames:        {frame_idx}")
            print(f"Detected refresh:    {self.refresh_rate} Hz")
            print(f"Measured refresh:    {measured_fps:.2f} Hz")
            print(f"Expected frame dt:   {expected_dt * 1000:.3f} ms")
            print(f"Max frame dt:        {max_frame_dt * 1000:.3f} ms")
            print(f"Dropped frames:      {dropped_frames}  ({drop_rate:.3f} %)")
            if self.saved_filename:
                print(f"Saved file:          {self.saved_filename}")
            print("=========================================================", flush=True)


def make_config_from_main_settings() -> Stage1Config:
    """Build config from edit-in-file settings for convenient IDE runs."""
    # ================== Frequently edited run settings ==================
    ONLINE_MODE = False
    FS = 2000
    PORT = "COM7"
    REQUIRED_REFRESH_RATE = 600
    RANDOM_SEED = None

    # ================== Stimulus and trial settings ==================
    STIM_FREQ = 10.0
    PHASE_COUNT = 6
    PHASE_STEP_DEG = None
    INITIAL_PHASE_DEG = 0.0
    BLOCK_COUNT = 1
    TRIALS_PER_PHASE_PER_BLOCK = 10

    SQUARE_SCALE = 0.22
    STIMULUS_DURATION_SEC = 2.5
    ITI_MIN_SEC = 0.8
    ITI_MAX_SEC = 1.2
    BLOCK_BREAK_SEC = 10.0

    TRIGGER_BASE = 1
    SAVE_PREFIX = "Stage1_SSVEP_PhaseTransfer"
    WAVEFORM = "square"

    return Stage1Config(
        online_mode=ONLINE_MODE,
        fs=FS,
        port=PORT,
        required_refresh_rate=REQUIRED_REFRESH_RATE,
        random_seed=RANDOM_SEED,
        stim_freq=STIM_FREQ,
        phase_count=PHASE_COUNT,
        phase_step_deg=PHASE_STEP_DEG,
        initial_phase_deg=INITIAL_PHASE_DEG,
        block_count=BLOCK_COUNT,
        trials_per_phase_per_block=TRIALS_PER_PHASE_PER_BLOCK,
        square_scale=SQUARE_SCALE,
        stimulus_duration_sec=STIMULUS_DURATION_SEC,
        iti_min_sec=ITI_MIN_SEC,
        iti_max_sec=ITI_MAX_SEC,
        block_break_sec=BLOCK_BREAK_SEC,
        trigger_base=TRIGGER_BASE,
        save_prefix=SAVE_PREFIX,
        waveform=WAVEFORM,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage1 SSVEP phase-transfer experiment.")
    parser.add_argument("--online", action="store_true", help="Enable EEG acquisition and serial triggers.")
    parser.add_argument("--fs", type=int, default=2000)
    parser.add_argument("--port", type=str, default="COM7")
    parser.add_argument("--required-refresh-rate", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=None)

    parser.add_argument("--stim-freq", type=float, default=10.0)
    parser.add_argument("--phase-count", type=int, default=6)
    parser.add_argument("--phase-step-deg", type=float, default=None)
    parser.add_argument("--initial-phase-deg", type=float, default=0.0)
    parser.add_argument("--block-count", type=int, default=1)
    parser.add_argument("--trials-per-phase-per-block", type=int, default=10)

    parser.add_argument("--square-scale", type=float, default=0.22)
    parser.add_argument("--stimulus-duration-sec", type=float, default=2.5)
    parser.add_argument("--iti-min-sec", type=float, default=0.8)
    parser.add_argument("--iti-max-sec", type=float, default=1.2)
    parser.add_argument("--block-break-sec", type=float, default=10.0)

    parser.add_argument("--trigger-base", type=int, default=1)
    parser.add_argument("--save-prefix", type=str, default="Stage1_SSVEP_PhaseTransfer")
    parser.add_argument("--waveform", choices=("square", "sine"), default="square")
    return parser.parse_args()


def make_config_from_args(args: argparse.Namespace) -> Stage1Config:
    return Stage1Config(
        online_mode=args.online,
        fs=args.fs,
        port=args.port,
        required_refresh_rate=args.required_refresh_rate,
        random_seed=args.random_seed,
        stim_freq=args.stim_freq,
        phase_count=args.phase_count,
        phase_step_deg=args.phase_step_deg,
        initial_phase_deg=args.initial_phase_deg,
        block_count=args.block_count,
        trials_per_phase_per_block=args.trials_per_phase_per_block,
        square_scale=args.square_scale,
        stimulus_duration_sec=args.stimulus_duration_sec,
        iti_min_sec=args.iti_min_sec,
        iti_max_sec=args.iti_max_sec,
        block_break_sec=args.block_break_sec,
        trigger_base=args.trigger_base,
        save_prefix=args.save_prefix,
        waveform=args.waveform,
    )


def main() -> None:
    # Keep this False for one-click IDE runs. Set True only when you want to
    # override settings from the terminal, e.g. --online --port COM7.
    USE_COMMAND_LINE_ARGS = False

    if USE_COMMAND_LINE_ARGS:
        config = make_config_from_args(parse_args())
    else:
        config = make_config_from_main_settings()

    experiment = Stage1PhaseTransferExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
