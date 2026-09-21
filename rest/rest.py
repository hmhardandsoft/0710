#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:30:00 2026

@author: computer
"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QTimer, QThread
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
        if not self.valid: return
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
class DataAcquisitionThread(QThread):
    def __init__(self, fs=2000, port='COM8', online_mode=True, parent=None):
        super().__init__(parent)
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
            self.eeg_dev.set_frequency(self.fs) # 动态设置采样率
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
                        self.eeg_data = np.append(self.eeg_data, frames, axis=0)
                except Exception as e:
                    print(f"Data acquisition error: {str(e)}")
                    break
            else:
                time.sleep(0.01)

    def send_trigger(self, value):
        if self.online_mode and self.trigger:
            self.trigger.setData(value)
        else:
            print(f"[Trigger Demo] Sending trigger: {value}")

    def stop(self, filename_prefix="RestingState"):
        self.running = False
        if self.online_mode:
            if len(self.eeg_data) > 0:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'{filename_prefix}_{timestamp}.mat'
                    savemat(filename, {'data': self.eeg_data, 'Fs': self.fs})
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
        self.wait()

# ================== 静息态主界面 ==================
class RestingStateExperiment(QMainWindow):
    def __init__(self, online_mode=True, run_duration=60):
        super().__init__()
        self.online_mode = online_mode
        self.run_duration = run_duration # 单次 Run 时长 (秒)
        self.is_closing = False
        
        self.experiment_started = False
        self.experiment_finished = False
        self.start_time = 0.0
        
        # 初始化采集线程 (匹配 iRecorder W8: 2000Hz)
        self.data_thread = DataAcquisitionThread(fs=2000, port='COM8', online_mode=self.online_mode)
        
        self.init_ui()
        self.data_thread.start()

    def init_ui(self):
        mode_str = "ONLINE" if self.online_mode else "OFFLINE"
        self.setWindowTitle(f'Resting State EEG - {mode_str}')
        self.showFullScreen()
        self.setStyleSheet("background-color: black;")
        self.setCursor(Qt.BlankCursor)

        screen = QApplication.primaryScreen().geometry()
        self.width = screen.width()
        self.height = screen.height()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_experiment)
        self.timer.setInterval(100) # 静息态不需要高频刷新UI，100ms即可

    def start_experiment(self):
        self.experiment_started = True
        self.start_time = time.time()
        self.data_thread.send_trigger(100) # 实验开始打标: 100
        self.timer.start()
        self.update()
        print(f"Recording started for {self.run_duration} seconds...")

    def update_experiment(self):
        if not self.experiment_started: return
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.run_duration:
            self.finish_experiment()

    def finish_experiment(self):
        self.timer.stop()
        # self.data_thread.send_trigger(11) 
        self.experiment_finished = True
        self.experiment_started = False
        self.update()
        print("Experiment Finished.")
        QTimer.singleShot(2000, self.close_application)

    def close_application(self):
        if self.is_closing: 
            return
        self.is_closing = True
        print("Stopping and saving data...")
        if self.timer.isActive(): self.timer.stop()
        self.experiment_started = False
        if self.data_thread.isRunning():
            self.data_thread.stop(filename_prefix="Resting_1min")
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.contentsRect()

        if not self.experiment_started and not self.experiment_finished:
            painter.setPen(Qt.white)
            painter.setFont(QFont('Arial', 24))
            text = "按空格键开始静息态采集 (注视屏幕中央)" + ("\n(Offline Test Mode)" if not self.online_mode else "")
            painter.drawText(rect, Qt.AlignCenter, text)
        elif self.experiment_started:
            # 绘制屏幕中央的注视十字
            pen = QPen(Qt.white, 8)
            painter.setPen(pen)
            cx, cy = self.width // 2, self.height // 2
            length = 50
            painter.drawLine(cx - length, cy, cx + length, cy)
            painter.drawLine(cx, cy - length, cx, cy + length)
        elif self.experiment_finished:
            painter.setPen(Qt.white)
            painter.setFont(QFont('Arial', 24))
            painter.drawText(rect, Qt.AlignCenter, "本轮静息态采集结束")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not self.experiment_started and not self.experiment_finished:
            self.start_experiment()
        elif event.key() == Qt.Key_Escape:
            self.close_application()

    def closeEvent(self, event):
        self.close_application()
        event.accept()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ENABLE_ONLINE_MODE = True
    window = RestingStateExperiment(online_mode=ENABLE_ONLINE_MODE, run_duration=60)
    window.show()
    sys.exit(app.exec_())