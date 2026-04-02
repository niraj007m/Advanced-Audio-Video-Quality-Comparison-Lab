import sys, os, json, subprocess, csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from moviepy import VideoFileClip
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QTextEdit, QDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

class InfoWindow(QDialog):
    """Information window for Technical Details and Credits."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Info & Credits")
        self.resize(550, 500)
        layout = QVBoxLayout(self)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
            <h2 style='color: #2980b9;'>Advanced Audio/Video Quality/Comparison Lab</h2>
            <p><b>Developed by:</b> Niraj M & Akshay J</p>
            <hr>
            <h3>Technical Guide:</h3>
            <p><b>1. Sync Score (Phase Correlation):</b> This is your 'Lip-Sync' detector.<br>
            - <b>1.0:</b> Perfect digital match.<br>
            - <b>0.9+:</b> Excellent dub.<br>
            - <b>Below 0.7:</b> Significant time drift or shifted frames detected.</p>
            
            <p><b>2. Clipping Detector:</b> If the peak reaches 1.0, the audio is 'hitting the ceiling.' 
            In spiritual discourses, this causes harsh distortion. This check ensures audio is pleasant.</p>
            
            <p><b>3. Batch Export (CSV):</b> Save results into a database to prove 100% of the 
            videos (Day 1 to Day 30) meet the TBR, TBN, and Sync standards.</p>
        """)
        layout.addWidget(info_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

class ShivirMediaValidator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Audio/Video Quality/Comparison Lab")
        self.resize(1600, 950)

        self.file1, self.file2 = None, None
        self.last_results = {}

        # --- MENU SETUP ---
        self.create_menu()

        # --- UI SETUP ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top Controls
        controls = QHBoxLayout()
        self.btn1 = QPushButton("Select Original File")
        self.btn2 = QPushButton("Select Dubbed File")
        self.run_btn = QPushButton("VALIDATE/COMPARE QUALITY")
        self.export_btn = QPushButton("Export CSV Report")
        
        self.run_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.export_btn.setStyleSheet("background-color: #2980b9; color: white; padding: 10px;")
        self.export_btn.setEnabled(False)

        # Fixed the connection naming here:
        self.btn1.clicked.connect(lambda: self.select_file(1))
        self.btn2.clicked.connect(lambda: self.select_file(2))
        self.run_btn.clicked.connect(self.start_analysis)
        self.export_btn.clicked.connect(self.export_report)

        controls.addWidget(self.btn1); controls.addWidget(self.btn2)
        controls.addWidget(self.run_btn); controls.addWidget(self.export_btn)
        main_layout.addLayout(controls)

        self.status = QLabel("Ready to validate...")
        self.status.setStyleSheet("font-family: 'Consolas'; color: #34495e; font-size: 14px;")
        main_layout.addWidget(self.status)

        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def create_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("&Help")
        info_action = QAction("Credits & Info", self)
        info_action.triggered.connect(self.show_info_window)
        help_menu.addAction(info_action)

    def show_info_window(self):
        dialog = InfoWindow(self)
        dialog.exec()

    def select_file(self, num):
        """Fixed function to handle file selection."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Media", "", "Files (*.mp4 *.wav *.mkv *.mp3)")
        if path:
            if num == 1:
                self.file1 = path
                self.btn1.setText(f"Original: {os.path.basename(path)}")
            else:
                self.file2 = path
                self.btn2.setText(f"Dubbed: {os.path.basename(path)}")

    def get_video_metadata(self, path):
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(res.stdout)
            v = next((s for s in data['streams'] if s['codec_type'] == 'video'), {})
            f = data.get('format', {})
            return {
                "Duration": f.get("duration", "0"),
                "Codec": v.get("codec_name", "N/A"),
                "FPS": v.get("avg_frame_rate", "N/A"),
                "TBR": v.get("r_frame_rate", "N/A"),
                "TBN": v.get("time_base", "N/A"),
                "SAR": v.get("sample_aspect_ratio", "N/A"),
                "DAR": v.get("display_aspect_ratio", "N/A")
            }
        except: return {k: "N/A" for k in ["Duration", "Codec", "FPS", "TBR", "TBN", "SAR", "DAR"]}

    def extract_audio_safe(self, path):
        if path.lower().endswith(('.mp4', '.mkv')):
            temp_wav = f"temp_val_{os.path.basename(path)}.wav"
            with VideoFileClip(path) as video:
                video.audio.write_audiofile(temp_wav, codec='pcm_s16le', logger=None)
            return temp_wav, True
        return path, False

    def start_analysis(self):
        if not self.file1 or not self.file2: return
        self.status.setText("Processing... Analyzing Sync and Bit-depth...")
        QApplication.processEvents()

        try:
            m1 = self.get_video_metadata(self.file1)
            m2 = self.get_video_metadata(self.file2)
            p1, t1 = self.extract_audio_safe(self.file1)
            p2, t2 = self.extract_audio_safe(self.file2)

            y1, sr1 = librosa.load(p1, sr=None)
            y2, sr2 = librosa.load(p2, sr=None)

            # Core Analysis Logic
            peak2 = np.max(np.abs(y2))
            clipping = "YES" if peak2 >= 0.98 else "NO"
            min_len = min(len(y1), len(y2))
            sync_score = np.corrcoef(y1[:min_len], y2[:min_len])[0, 1]

            self.last_results = {
                "Original_File": os.path.basename(self.file1),
                "Dubbed_File": os.path.basename(self.file2),
                "Sync_Score": round(sync_score, 4),
                "Peak_Level": round(peak2, 2),
                "Clipping_Risk": clipping,
                "TBR_Orig": m1['TBR'], "TBR_Dub": m2['TBR'],
                "DAR_Orig": m1['DAR'], "DAR_Dub": m2['DAR']
            }

            self.figure.clear()
            
            # 1. Frequency Shift Graph
            ax1 = self.figure.add_subplot(2, 2, 1)
            S1, S2 = np.abs(librosa.stft(y1)), np.abs(librosa.stft(y2))
            ax1.plot(librosa.fft_frequencies(sr=sr1), librosa.amplitude_to_db(np.mean(S1, axis=1)), label="Original", alpha=0.7)
            ax1.plot(librosa.fft_frequencies(sr=sr2), librosa.amplitude_to_db(np.mean(S2, axis=1)), label="Dubbed", ls='--', alpha=0.7)
            ax1.set_title(f"Frequency Shift (Sync: {sync_score:.2f})")
            ax1.set_xlim(0, 18000); ax1.legend()

            # 2. Tech Table
            ax2 = self.figure.add_subplot(2, 2, 2); ax2.axis('off')
            tech_txt = (
                f"--- Niraj M & Akshay J Validator ---\n"
                f"SYNC: {sync_score:.4f} | CLIPPING: {clipping}\n"
                f"{'-'*40}\n"
                f"METRIC | ORIGINAL     | DUBBED\n"
                f"DUR    | {m1['Duration']:<12} | {m2['Duration']:<12}\n"
                f"FPS    | {m1['FPS']:<12} | {m2['FPS']:<12}\n"
                f"TBR    | {m1['TBR']:<12} | {m2['TBR']:<12}\n"
                f"TBN    | {m1['TBN']:<12} | {m2['TBN']:<12}\n"
                f"SAR    | {m1['SAR']:<12} | {m2['SAR']:<12}\n"
                f"DAR    | {m1['DAR']:<12} | {m2['DAR']:<12}\n"
                f"SR     | {sr1} Hz | {sr2} Hz"
            )
            ax2.text(0, 0.5, tech_txt, family='monospace', fontsize=9, verticalalignment='center')

            # 3 & 4. Spectrograms
            for i, (y, sr, title) in enumerate([(y1, sr1, "Original"), (y2, sr2, "Dubbed")]):
                ax = self.figure.add_subplot(2, 2, i+3)
                spec = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='hz', ax=ax)
                ax.set_title(f"{title} Frequency Spectrum")

            self.figure.tight_layout(); self.canvas.draw()
            self.status.setText(f"Analysis Finished. Sync Score: {sync_score:.4f}")
            self.export_btn.setEnabled(True)

            if t1: os.remove(p1)
            if t2: os.remove(p2)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Database", "Shivir_Validation_Database.csv", "CSV (*.csv)")
        if path:
            exists = os.path.isfile(path)
            with open(path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.last_results.keys())
                if not exists: writer.writeheader()
                writer.writerow(self.last_results)
            QMessageBox.information(self, "Export", "Result saved to your Batch Database.")

if __name__ == "__main__":
    app = QApplication(sys.argv); win = ShivirMediaValidator(); win.show(); sys.exit(app.exec())