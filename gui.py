from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import QObject, pyqtSignal
import subprocess
import threading
import signal
import requests
import time
import ctypes
import multiprocessing
import cmath
import mmap
import pyexpat
import json
import sys
import os

dirname = os.path.dirname(__file__)

process = None

signal.signal(signal.SIGINT, signal.SIG_DFL)

class ConsoleSignals(QObject):
    closed = pyqtSignal()
    console_update = pyqtSignal(str)
    
class ConsoleWindow(QMainWindow):
    MAX_TEXT_LENGTH = 100000

    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_text_length = 0
        
    def initUI(self):
        self.setWindowTitle("Img Diffuse")
        self.resize(700, 400)
        screen_geometry = QApplication.primaryScreen().geometry()
        center_x = (screen_geometry.width() - self.width()) // 2
        center_y = (screen_geometry.height() - self.height()) // 2
        self.move(center_x, center_y)
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet("background-color: black; color: white;")
        self.setCentralWidget(self.text_edit)
        
    def write(self, text):
        if "%" in text:
            self.text_edit.undo()
        else:
            self.current_text_length += len(text)
            if self.current_text_length > self.MAX_TEXT_LENGTH:
                self.text_edit.clear()
                self.current_text_length = 0
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.insertPlainText(text)
        self.text_edit.ensureCursorVisible()
        
def redirect_console(signals):
    global process
    main_path = os.path.join(dirname, "main.py")
    if "_internal" in dirname: 
        main_path = os.path.join(dirname, "../main.exe")
    if "Frameworks" in dirname:
        main_path = os.path.normpath(os.path.join(dirname, "../MacOS/main"))
    if main_path.endswith(".py"):
        command = ["python3", "main.py"]
    else:
        command = [main_path]
    process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            signals.console_update.emit(line)
    process.wait()

def kill_server():
    global server_thread
    host = None
    port = None
    config_path = os.path.normpath(os.path.join(dirname, "config.json"))
    if os.path.exists(config_path):
        with open(config_path) as config:
            data = json.load(config)
            host = data["host"]
            port = data["port"]
    try:
        requests.get(f"http://{host}:{port}/shutdown")
    except:
        pass

def kill_process():
    kill_server()
    os.kill(process.pid, signal.SIGINT)
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    console_window = ConsoleWindow()
    console_window.show()
    signals = ConsoleSignals()
    signals.console_update.connect(console_window.write)
    kill_server()
    time.sleep(0.3)
    thread = threading.Thread(target=redirect_console, args=(signals,))
    thread.start()
    app.aboutToQuit.connect(kill_process)
    ret = app.exec()
    kill_process()
    sys.exit(ret)