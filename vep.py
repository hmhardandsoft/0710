#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:23:01 2026

@author: computer
"""

import sys
import time
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF,QThread
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
                        self.eeg_data = np.append(self.eeg_data, frames, axis=0)
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

    def stop(self, filename_prefix="VEP"):
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


# ================== 主界面 ==================
class TransientVEP(QMainWindow):
    def __init__(self, online_mode=True, block_duration=60):
        super().__init__()
        self.online_mode = online_mode
        self.block_duration = block_duration # 单个 Block 持续时间 (秒)
        
        self.experiment_started = False
        self.experiment_finished = False
        
        # 时序控制变量
        self.block_start_time = 0.0
        self.current_onset_time = 0.0
        self.current_interval_sec = 0.0 # 当前随机出的 SOA (Onset到下一个Onset的时间)
        
        self.is_highlight = False # 当前屏幕是否处于 10ms 的高亮状态
        self.highlight_duration = 0.010 # 固定的 10ms 高亮时间
        
        # 视觉参数
        self.block_size = 350
        self.color_val = 0 
        
        # 初始化采集线程 
        self.data_thread = DataAcquisitionThread(fs=2000, port='COM8', online_mode=self.online_mode)
        
        self.init_ui()
        self.data_thread.start()
        
        self.is_closing = False

    def init_ui(self):
        mode_str = "ONLINE" if self.online_mode else "OFFLINE"
        self.setWindowTitle(f'Transient VEP - {mode_str}')
        self.showFullScreen()
        self.setStyleSheet("background-color: black;")
        self.setCursor(Qt.BlankCursor)

        screen = QApplication.primaryScreen().geometry()
        self.width = screen.width()
        self.height = screen.height()
        
        self.rect = QRectF(
            (self.width - self.block_size) / 2, 
            (self.height - self.block_size) / 2, 
            self.block_size, 
            self.block_size
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_stimulation)
        # 极小间隔轮询，保证时间判定的高精度 (实际刷新最终受到屏幕硬件制约)
        self.timer.setInterval(1) 

    def start_experiment(self):
        self.experiment_started = True
        self.block_start_time = time.time()
        
        # 初始化第一个脉冲
        self.trigger_pulse()
        self.timer.start()

    def trigger_pulse(self):
        """触发单次 10ms 的高亮脉冲"""
        self.current_onset_time = time.time()
        # 随机选取 50ms 到 125ms 之间的连续值作为本轮的 Onset 间隔
        self.current_interval_sec = random.uniform(0.050, 0.125) 
        
        # 触发打标 (标签设定为 1，代表 Transient Onset)
        self.data_thread.send_trigger(1)
        
        self.is_highlight = True
        self.color_val = 255
        self.update()

    def update_stimulation(self):
        if not self.experiment_started: return

        now = time.time()
        elapsed_block = now - self.block_start_time
        elapsed_pulse = now - self.current_onset_time
        
        # 1. 检查是否达到了单个 Block (1分钟) 的总时间
        if elapsed_block >= self.block_duration:
            self.finish_experiment()
            return
            
        # 2. 控制 10ms 的高亮时长 (转为纯黑)
        if self.is_highlight and elapsed_pulse >= self.highlight_duration:
            self.is_highlight = False
            self.color_val = 0
            self.update()
            
        # 3. 控制随机 Interval (Onset 间隔) 以衔接下一次脉冲
        if elapsed_pulse >= self.current_interval_sec:
            self.trigger_pulse()

    def finish_experiment(self):
        self.timer.stop()
        self.experiment_finished = True
        self.experiment_started = False
        self.color_val = 0
        self.update()
        print("Block Finished. All random pulses completed.")
        QTimer.singleShot(2000, self.close_application)

    def close_application(self):
        if self.is_closing:  # 新增：如果已经在关闭中了，就直接 return
            return
        self.is_closing = True
        print("Stopping experiment safely...")
        if self.timer.isActive(): self.timer.stop()
        self.experiment_started = False
        if self.data_thread.isRunning():
            self.data_thread.stop(filename_prefix="TransientVEP_1min")
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        if not self.experiment_started:
            painter.setPen(Qt.white)
            font = QFont('Arial', 24)
            painter.setFont(font)
            text = "按空格键开始瞬态VEP实验 (1 min)" + ("\n(Offline Test Mode)" if not self.online_mode else "")
            if self.experiment_finished:
                text = "实验结束"
            painter.drawText(self.contentsRect(), Qt.AlignCenter, text)
        else:
            
            color = QColor(self.color_val, self.color_val, self.color_val)
            painter.fillRect(self.rect, color)
            
           
            if not self.online_mode:
                painter.setPen(Qt.gray)
                painter.drawText(20, 30, f"Block Time: {time.time() - self.block_start_time:.1f} / {self.block_duration} s")
                painter.drawText(20, 50, f"Current SOA: {self.current_interval_sec*1000:.1f} ms")

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
    ENABLE_ONLINE_MODE = False 
    window = TransientVEP(online_mode=ENABLE_ONLINE_MODE, block_duration=60)
    window.show()
    sys.exit(app.exec_())