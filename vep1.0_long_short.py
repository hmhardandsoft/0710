#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:23:01 2026

@author: computer
"""

import sys
import time
import random
import threading
import numpy as np
import glfw
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


# 尝试导入硬件库
try:
    from eConEXG import iRecorder
    import serial
    HARDWARE_LIBS_AVAILABLE = True
except ImportError:
    HARDWARE_LIBS_AVAILABLE = False
    print("Warning: Hardware libraries (eConEXG/serial) not found. Forcing Offline Mode.")


# ================== 硬件接口类 (串口打标) ==================
class Niantong_port:
    def __init__(self, port_addr, baudrate=115200):
        self.valid = False
        try:
            if HARDWARE_LIBS_AVAILABLE:
                self.port = serial.Serial(port=port_addr, baudrate=baudrate)
                self.valid = True
        except Exception as e:
            print(f"Serial Port Error: {e}")

    def setData(self, label):
        if not self.valid:
            return
        if label < 1 or label > 255:
            print("Error: Label must be in range 1 to 255.")
            return

        label_hex = format(label, '02X')
        frame_end = '55660D'
        data_to_send = label_hex + frame_end
        send_bytes = bytes.fromhex(data_to_send)
        try:
            self.port.write(send_bytes)
        except Exception as e:
            print(f"Serial Write Error: {e}")

    def stop(self):
        if self.valid and self.port.is_open:
            self.port.close()


# ================== 数据采集线程 ==================
class DataAcquisitionThread(threading.Thread):
    def __init__(self, fs=2000, port='COM7', online_mode=True):
        super().__init__(daemon=True)
        self.fs = fs
        self.port_name = port
        self.online_mode = online_mode and HARDWARE_LIBS_AVAILABLE
        self.running = True
        self.eeg_dev = None
        self.trigger = None
        self.eeg_data = np.empty((0, 9))

    def init_devices(self):
        if not self.online_mode:
            print("[System] Offline Mode: Skipping device initialization.")
            return True

        try:
            self.eeg_dev = iRecorder(dev_type="USB8")
            self.eeg_dev.set_frequency(self.fs)
            self.eeg_dev.find_devs()
            while True:
                ret = self.eeg_dev.get_devs()
                if ret:
                    break
                time.sleep(0.1)

            self.eeg_dev.connect_device(ret[0])
            self.eeg_dev.start_acquisition_data(with_q=True)

            self.trigger = Niantong_port(self.port_name)
            return True

        except Exception as e:
            print(f"Device initialization error: {str(e)}")
            return False

    def run(self):
        if not self.init_devices():
            return

        while self.running:
            if self.online_mode:
                try:
                    frames = self.eeg_dev.get_data(timeout=0.01)
                    if frames:
                        self.eeg_data = np.append(
                            self.eeg_data,
                            frames,
                            axis=0
                        )
                except Exception as e:
                    print(f"Data acquisition error: {str(e)}")
                    break
            else:
                time.sleep(0.01)

    def send_trigger(self, value):
        if self.online_mode and self.trigger:
            self.trigger.setData(value)

        # else:
        #     print(f"[Trigger Demo] Sending trigger: {value}")

    def stop(self, filename_prefix="VEP", filename_suffix=None):
        self.running = False

        if self.online_mode:
            if len(self.eeg_data) > 0:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    if filename_suffix:
                        filename = (
                            f'{filename_prefix}_{timestamp}_{filename_suffix}.mat'
                        )
                    else:
                        filename = f'{filename_prefix}_{timestamp}.mat'

                    savemat(
                        filename,
                        {
                            'data': self.eeg_data,
                            'Fs': self.fs
                        }
                    )

                    print(f"Data saved to {filename}")

                except Exception as e:
                    print(f"Data saving error: {str(e)}")

            if self.eeg_dev:
                try:
                    self.eeg_dev.stop_acquisition()
                    self.eeg_dev.close_dev()
                except:
                    pass

            if self.trigger:
                self.trigger.stop()

        if threading.current_thread() is not self:
            self.join(timeout=5.0)


# ================== 主界面 ==================
class TransientVEP:
    def __init__(
        self,
        online_mode=True,
        block_duration=60,
        required_refresh_rate=None,
        soa_mode="long",
        square_scale=0.22,
    ):
        self.online_mode = online_mode
        self.block_duration = block_duration  # 单个 Block 持续时间 (秒)
        self.required_refresh_rate = required_refresh_rate

        # SOA 模式：long / short
        self.soa_mode = str(soa_mode).strip().lower()

        if self.soa_mode == "long":
            self.soa_min_sec = 1.100
            self.soa_max_sec = 2.100

        elif self.soa_mode == "short":
            self.soa_min_sec = 0.060
            self.soa_max_sec = 0.135

        else:
            raise ValueError(
                f"Invalid soa_mode={soa_mode!r}. "
                "Please use 'long' or 'short'."
            )

        # 时序控制变量：以显示帧为单位控制 onset / offset
        self.frame_idx = 0
        self.onset_frame = 0
        self.offset_frame = 0
        self.next_onset_frame = 0
        self.current_soa_frames = 0
        self.trigger_sent_for_onset = False
        self.highlight_duration = 0.010  # 固定的 10ms 高亮时间
        self.debug_drop_print_limit = 5

        # ================== 视觉参数 ==================
        # 刺激方块边长 = 屏幕短边 × square_scale
        self.square_scale = square_scale

        # 初始化采集线程
        self.data_thread = DataAcquisitionThread(
            fs=2000,
            port='COM7',
            online_mode=self.online_mode
        )

        self.init_ui()
        self.data_thread.start()

        self.is_closing = False

    def init_ui(self):
        mode_str = "ONLINE" if self.online_mode else "OFFLINE"

        if not glfw.init():
            sys.exit("GLFW init failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)

        self.refresh_rate = int(mode.refresh_rate)
        self.width = int(mode.size.width)
        self.height = int(mode.size.height)

        # 根据屏幕短边和 square_scale 计算刺激方块大小
        self.block_size = (
            min(self.width, self.height)
            * self.square_scale
        )

        if (
            self.required_refresh_rate is not None
            and self.refresh_rate != self.required_refresh_rate
        ):
            print(
                f"[VEP] WARNING: expected "
                f"{self.required_refresh_rate} Hz, "
                f"detected {self.refresh_rate} Hz. Continuing.",
                flush=True,
            )

        glfw.window_hint(
            glfw.DOUBLEBUFFER,
            GL_TRUE
        )

        glfw.window_hint(
            glfw.SAMPLES,
            0
        )

        glfw.window_hint(
            glfw.REFRESH_RATE,
            self.refresh_rate
        )

        self.window = glfw.create_window(
            self.width,
            self.height,
            f"Transient VEP - {mode_str}",
            monitor,
            None,
        )

        if not self.window:
            glfw.terminate()
            sys.exit("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_input_mode(
            self.window,
            glfw.CURSOR,
            glfw.CURSOR_HIDDEN
        )

        glfw.set_key_callback(
            self.window,
            self._key_callback
        )

        glClearColor(
            0.0,
            0.0,
            0.0,
            1.0
        )

        glViewport(
            0,
            0,
            self.width,
            self.height
        )

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        glOrtho(
            -self.width / 2,
            self.width / 2,
            -self.height / 2,
            self.height / 2,
            -1.0,
            1.0
        )

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        self.stim_frames = max(
            1,
            int(
                round(
                    self.highlight_duration
                    * self.refresh_rate
                )
            )
        )

        self.total_frames = max(
            1,
            int(
                round(
                    self.block_duration
                    * self.refresh_rate
                )
            )
        )

        self.soa_min_frames = max(
            self.stim_frames + 1,
            int(
                round(
                    self.soa_min_sec
                    * self.refresh_rate
                )
            )
        )

        self.soa_max_frames = max(
            self.stim_frames + 1,
            int(
                round(
                    self.soa_max_sec
                    * self.refresh_rate
                )
            )
        )

        expected_dt_ms = 1000.0 / self.refresh_rate

        print(
            f"[VEP] Display: "
            f"{self.width}x{self.height} "
            f"@ {self.refresh_rate} Hz "
            f"(expected frame={expected_dt_ms:.3f} ms)",
            flush=True,
        )

        print(
            f"[VEP] SOA mode: "
            f"{self.soa_mode.upper()} "
            f"({self.soa_min_sec:.3f}-"
            f"{self.soa_max_sec:.3f} s)",
            flush=True,
        )

        print(
            f"[VEP] Timing: highlight="
            f"{self.stim_frames} frames "
            f"({self.stim_frames / self.refresh_rate * 1000:.2f} ms), "
            f"SOA range="
            f"{self.soa_min_frames}-"
            f"{self.soa_max_frames} frames "
            f"({self.soa_min_frames / self.refresh_rate * 1000:.2f}-"
            f"{self.soa_max_frames / self.refresh_rate * 1000:.2f} ms), "
            f"block={self.total_frames} frames "
            f"({self.block_duration:.2f} s)",
            flush=True,
        )

    def _key_callback(
        self,
        window,
        key,
        scancode,
        action,
        mods
    ):
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(
                window,
                True
            )

    def _draw_square(self, intensity):
        half = self.block_size / 2.0

        glColor3f(
            float(intensity),
            float(intensity),
            float(intensity)
        )

        glBegin(GL_QUADS)

        glVertex2f(
            -half,
            -half
        )

        glVertex2f(
            half,
            -half
        )

        glVertex2f(
            half,
            half
        )

        glVertex2f(
            -half,
            half
        )

        glEnd()

    def _render_frame(self, highlight):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        self._draw_square(
            1.0 if highlight else 0.0
        )

        glFlush()

    def _wait_for_space(self):
        print(
            "[VEP] Press SPACE to start; ESC to quit.",
            flush=True
        )

        while not glfw.window_should_close(
            self.window
        ):
            if (
                glfw.get_key(
                    self.window,
                    glfw.KEY_SPACE
                )
                == glfw.PRESS
            ):
                return True

            glClear(
                GL_COLOR_BUFFER_BIT
            )

            glfw.swap_buffers(
                self.window
            )

            glfw.poll_events()
            time.sleep(0.01)

        return False

    def _schedule_pulse(self):
        self.onset_frame = self.frame_idx

        self.offset_frame = (
            self.onset_frame
            + self.stim_frames
        )

        soa_sec = random.uniform(
            self.soa_min_sec,
            self.soa_max_sec
        )

        self.current_soa_frames = max(
            self.stim_frames + 1,
            int(
                round(
                    soa_sec
                    * self.refresh_rate
                )
            )
        )

        self.next_onset_frame = (
            self.onset_frame
            + self.current_soa_frames
        )

        self.trigger_sent_for_onset = False

    def close_application(self):
        if self.is_closing:
            return

        self.is_closing = True

        print(
            "Stopping experiment safely..."
        )

        if self.data_thread.is_alive():
            self.data_thread.stop(
                filename_prefix="TransientVEP_1min",
                filename_suffix=self.soa_mode,
            )

        try:
            glfw.destroy_window(
                self.window
            )
        except Exception:
            pass

        glfw.terminate()

    def run(self):
        if not self._wait_for_space():
            self.close_application()
            return

        self.frame_idx = 0

        self._schedule_pulse()

        t_start = time.perf_counter()
        t_prev = t_start

        expected_dt = (
            1.0
            / self.refresh_rate
        )

        dropped_frames = 0
        max_frame_dt = 0.0

        try:
            while not glfw.window_should_close(
                self.window
            ):
                if self.frame_idx >= self.total_frames:
                    break

                if (
                    self.frame_idx
                    >= self.next_onset_frame
                ):
                    self._schedule_pulse()

                highlight = (
                    self.onset_frame
                    <= self.frame_idx
                    < self.offset_frame
                )

                if (
                    self.frame_idx
                    == self.onset_frame
                    and not self.trigger_sent_for_onset
                ):
                    self.data_thread.send_trigger(1)

                    self.trigger_sent_for_onset = True

                self._render_frame(
                    highlight
                )

                glfw.swap_buffers(
                    self.window
                )

                glfw.poll_events()

                t_now = time.perf_counter()
                dt = t_now - t_prev

                max_frame_dt = max(
                    max_frame_dt,
                    dt
                )

                if (
                    self.frame_idx > 5
                    and dt > expected_dt * 1.5
                ):
                    dropped_frames += 1

                    if (
                        dropped_frames
                        <= self.debug_drop_print_limit
                    ):
                        print(
                            f"[VEP] Dropped-frame warning: "
                            f"frame={self.frame_idx}, "
                            f"dt={dt * 1000:.3f} ms, "
                            f"threshold="
                            f"{expected_dt * 1.5 * 1000:.3f} ms",
                            flush=True,
                        )

                t_prev = t_now

                self.frame_idx += 1

        finally:
            t_end = time.perf_counter()

            glClear(
                GL_COLOR_BUFFER_BIT
            )

            glfw.swap_buffers(
                self.window
            )

            glfw.poll_events()

            time.sleep(1.0)

            self.close_application()

            stimulus_time = (
                t_end
                - t_start
            )

            measured_fps = (
                self.frame_idx
                / max(
                    stimulus_time,
                    1e-9
                )
            )

            drop_rate = (
                dropped_frames
                / max(
                    self.frame_idx,
                    1
                )
                * 100
            )

            print(
                "\n========== Transient VEP Summary =========="
            )

            print(
                f"Stimulus time:       "
                f"{stimulus_time:.2f} s"
            )

            print(
                f"Total frames:        "
                f"{self.frame_idx}"
            )

            print(
                f"Detected refresh:    "
                f"{self.refresh_rate} Hz"
            )

            print(
                f"Measured refresh:    "
                f"{measured_fps:.2f} Hz"
            )

            print(
                f"Expected frame dt:   "
                f"{expected_dt * 1000:.3f} ms"
            )

            print(
                f"Max frame dt:        "
                f"{max_frame_dt * 1000:.3f} ms"
            )

            print(
                f"Dropped frames:      "
                f"{dropped_frames}  "
                f"({drop_rate:.3f} %)"
            )

            print(
                "===========================================",
                flush=True
            )


if __name__ == '__main__':
    ENABLE_ONLINE_MODE = True

    # 选择 SOA 模式："long" 或 "short"
    SOA_MODE = "long"

    # ================== 刺激大小设置 ==================
    # 方块边长 = 屏幕短边 × SQUARE_SCALE
    SQUARE_SCALE = 0.22

    window = TransientVEP(
        online_mode=ENABLE_ONLINE_MODE,
        block_duration=60,
        soa_mode=SOA_MODE,
        square_scale=SQUARE_SCALE,
    )

    window.run()