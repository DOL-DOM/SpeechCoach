"""
main.py – Speech Coach 메인 GUI (PyQt5)
"""
import sys
import os
import json
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTabWidget,
    QTextEdit, QScrollArea, QFrame, QLineEdit, QMessageBox,
    QSplitter, QGroupBox, QGridLayout, QSlider, QComboBox,
    QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette, QImage, QIcon, QPainter, QBrush

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager

# 한글 폰트 설정
def _set_korean_font():
    korean_fonts = [
        "Malgun Gothic",   # Windows 기본
        "NanumGothic",
        "AppleGothic",
        "Gulim",
        "Dotum",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in korean_fonts:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()
import numpy as np

# ── 내부 모듈 ──────────────────────────────────────────────────────────────
from video_analyzer   import VideoAnalyzer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  스타일 시트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLE = """
QWidget { background:#0f0f1a; color:#e8e8f0; font-family:'Segoe UI',sans-serif; }
QMainWindow { background:#0f0f1a; }
QTabWidget::pane { border:1px solid #2a2a4a; background:#13132a; border-radius:8px; }
QTabBar::tab { background:#1a1a3a; color:#888; padding:10px 22px;
               border-radius:6px 6px 0 0; margin-right:3px; font-size:13px; }
QTabBar::tab:selected { background:#6c63ff; color:#fff; }
QTabBar::tab:hover:!selected { background:#252545; color:#aaa; }
QPushButton {
  background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6c63ff,stop:1 #a855f7);
  color:#fff; border:none; border-radius:8px; padding:10px 22px;
  font-size:14px; font-weight:600; }
QPushButton:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5a52dd,stop:1 #9333ea); }
QPushButton:disabled { background:#2a2a4a; color:#555; }
QPushButton#secondary {
  background:#1e1e3a; border:1px solid #3a3a5a; }
QPushButton#secondary:hover { background:#252545; }
QLineEdit {
  background:#1a1a30; border:1px solid #3a3a5a; border-radius:6px;
  padding:8px 12px; color:#e8e8f0; font-size:13px; }
QLineEdit:focus { border-color:#6c63ff; }
QProgressBar {
  background:#1a1a30; border:1px solid #3a3a5a; border-radius:8px;
  height:18px; text-align:center; color:#fff; font-size:11px; }
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
  stop:0 #6c63ff,stop:1 #a855f7); border-radius:8px; }
QScrollBar:vertical { background:#1a1a30; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#3a3a5a; border-radius:4px; min-height:30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QGroupBox {
  border:1px solid #2a2a4a; border-radius:8px; margin-top:12px;
  padding-top:8px; font-size:13px; color:#aaa; }
QGroupBox::title { subcontrol-origin:margin; left:12px; top:-6px;
  background:#0f0f1a; padding:0 6px; }
QComboBox { background:#1a1a30; border:1px solid #3a3a5a; border-radius:6px;
  padding:6px 12px; color:#e8e8f0; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#1a1a30; selection-background-color:#6c63ff; }
QLabel#score_big { font-size:42px; font-weight:700; color:#6c63ff; }
QLabel#score_label { font-size:12px; color:#888; }
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  분석 워커 스레드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AnalysisWorker(QThread):
    progress  = pyqtSignal(int)
    status    = pyqtSignal(str)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, video_path, whisper_model, context):
        super().__init__()
        self.video_path    = video_path
        self.whisper_model = whisper_model
        self.context       = context

    def run(self):
        import logging
        import subprocess, json, os
        flog = logging.getLogger("worker")
        flog.setLevel(logging.DEBUG)
        fh = logging.FileHandler("worker.log", encoding="utf-8")
        flog.addHandler(fh)

        try:
            flog.info("STEP 1: 비디오 분석 시작")
            self.status.emit("📹 영상 분석 중...")
            va = VideoAnalyzer()
            video_r = va.analyze(
                self.video_path,
                progress_callback=lambda p: self.progress.emit(p))
            flog.info(f"STEP 1 완료: {video_r.get('overall_nonverbal_score')}")

            flog.info("STEP 2: 오디오 분석 (subprocess)")
            self.status.emit("🎤 음성 분석 중...")
            self.progress.emit(65)

            import tempfile
            result_file = tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w", encoding="utf-8")
            result_path = result_file.name
            result_file.close()

            worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "analyze_audio_worker.py")
            proc = subprocess.run(
                [sys.executable, worker_script,
                 self.video_path, self.whisper_model, result_path],
                timeout=600)

            flog.info(f"STEP 2 subprocess returncode={proc.returncode}")

            if not os.path.exists(result_path) or os.path.getsize(result_path) < 5:
                raise RuntimeError("오디오 분석 결과 파일이 없거나 비어있습니다.")

            with open(result_path, "r", encoding="utf-8") as f:
                audio_r = json.load(f)
            os.remove(result_path)
            flog.info(f"STEP 2 완료: verbal={audio_r.get('overall_verbal_score')}")

            self.progress.emit(100)
            self.finished.emit({
                "video":    video_r,
                "audio":    audio_r,
                "feedback": {}
            })
            flog.info("완료!")

        except Exception as e:
            import traceback
            msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            flog.error(msg)
            self.error.emit(msg)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 윈도우
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ SpeechCoach AI")
        self.setMinimumSize(720, 560)
        self.video_path = ""
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(20)

        # ── 헤더 ────────────────────────────────────────────────
        hdr = QLabel("🎙️ SpeechCoach AI")
        hdr.setFont(QFont("Segoe UI", 22, QFont.Bold))
        hdr.setStyleSheet("color:#6c63ff;")
        sub = QLabel("Multimodal 스피치·면접 교정 시스템")
        sub.setStyleSheet("color:#888; font-size:13px;")
        root.addWidget(hdr)
        root.addWidget(sub)

        # ── 영상 업로드 ─────────────────────────────────────────
        upload_group = QGroupBox("1. 영상 업로드")
        ugl = QHBoxLayout(upload_group)
        self.file_label = QLabel("파일을 선택하세요 (.mp4, .mov, .avi, .mkv)")
        self.file_label.setStyleSheet("color:#888; font-size:13px;")
        self.file_label.setWordWrap(True)
        btn_upload = QPushButton("📂 파일 선택")
        btn_upload.setObjectName("secondary")
        btn_upload.setFixedWidth(130)
        btn_upload.clicked.connect(self._select_file)
        ugl.addWidget(self.file_label, 1)
        ugl.addWidget(btn_upload)
        root.addWidget(upload_group)

        # ── 설정 ────────────────────────────────────────────────
        cfg_group = QGroupBox("2. 분석 설정")
        cgl = QGridLayout(cfg_group)

        cgl.addWidget(QLabel("Whisper 모델:"), 0, 0)
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["tiny", "base", "small", "medium"])
        self.whisper_combo.setCurrentText("base")
        cgl.addWidget(self.whisper_combo, 0, 1)

        cgl.addWidget(QLabel("분석 맥락:"), 1, 0)
        self.ctx_combo = QComboBox()
        self.ctx_combo.addItems(["면접", "발표/프레젠테이션", "토론", "스피치대회"])
        cgl.addWidget(self.ctx_combo, 1, 1)

        root.addWidget(cfg_group)

        # ── 진행 상황 ───────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#888; font-size:12px;")
        root.addWidget(self.progress_bar)
        root.addWidget(self.status_label)

        # ── 분석 버튼 ───────────────────────────────────────────
        self.btn_analyze = QPushButton("🚀 분석 시작")
        self.btn_analyze.setFixedHeight(48)
        self.btn_analyze.clicked.connect(self._start_analysis)
        root.addWidget(self.btn_analyze)
        root.addStretch()

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.flv)")
        if path:
            self.video_path = path
            name = Path(path).name
            self.file_label.setText(f"✅ {name}")
            self.file_label.setStyleSheet("color:#7fff7f; font-size:13px;")

    def _start_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "오류", "영상 파일을 먼저 선택하세요.")
            return

        self.btn_analyze.setEnabled(False)
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(
            self.video_path,
            self.whisper_combo.currentText(),
            self.ctx_combo.currentText()
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, results):
        self.btn_analyze.setEnabled(True)
        self.status_label.setText("✅ 분석 완료!")
        self.results_win = ResultsWindow(results, self)
        self.results_win.show()

    def _on_error(self, msg):
        self.btn_analyze.setEnabled(True)
        self.status_label.setText("❌ 오류 발생")
        QMessageBox.critical(self, "분석 오류", msg[:800])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  결과 윈도우
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ResultsWindow(QMainWindow):
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("📊 분석 결과 — SpeechCoach AI")
        self.setMinimumSize(1100, 780)
        self.setStyleSheet(STYLE)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)

        # ── 상단: 점수 배너 ─────────────────────────────────────
        root.addWidget(self._score_banner())

        # ── 탭 ──────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.addTab(self._tab_nonverbal(), "👁 비언어 분석")
        tabs.addTab(self._tab_verbal(),    "🎤 언어 분석")
        tabs.addTab(self._tab_feedback(),  "🤖 AI 피드백")
        tabs.addTab(self._tab_transcript(),"📝 전사 내용")
        root.addWidget(tabs)

        # ── 저장 버튼 ───────────────────────────────────────────
        btn_save = QPushButton("💾 결과 저장 (JSON)")
        btn_save.setObjectName("secondary")
        btn_save.setFixedWidth(200)
        btn_save.clicked.connect(self._save_results)
        hb = QHBoxLayout()
        hb.addStretch()
        hb.addWidget(btn_save)
        root.addLayout(hb)

    # ─────────────────────────────────────────────────────────────
    #  점수 배너
    # ─────────────────────────────────────────────────────────────
    def _score_banner(self):
        fb = self.results.get("feedback", {})
        vr = self.results.get("video", {})
        ar = self.results.get("audio", {})

        scores = fb.get("scores", {})
        nv = scores.get("nonverbal", vr.get("overall_nonverbal_score", 5))
        vb = scores.get("verbal",    ar.get("overall_verbal_score", 5))
        ov = scores.get("overall",   round((nv + vb) / 2, 1))

        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#13132a;border:1px solid #2a2a4a;"
            "border-radius:12px;padding:8px;}")
        lay = QHBoxLayout(frame)
        lay.setSpacing(40)

        def _score_widget(label, score, color="#6c63ff"):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setAlignment(Qt.AlignCenter)
            big = QLabel(f"{score:.1f}")
            big.setObjectName("score_big")
            big.setStyleSheet(f"font-size:40px;font-weight:700;color:{color};")
            big.setAlignment(Qt.AlignCenter)
            lbl = QLabel(f"{label}\n/ 10")
            lbl.setObjectName("score_label")
            lbl.setAlignment(Qt.AlignCenter)
            bar = QProgressBar()
            bar.setValue(int(score * 10))
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar{{background:#1a1a30;border-radius:4px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:4px;}}")
            l.addWidget(big)
            l.addWidget(lbl)
            l.addWidget(bar)
            return w

        lay.addWidget(_score_widget("비언어", nv, "#6c63ff"))
        lay.addWidget(_score_widget("언어",   vb, "#a855f7"))
        lay.addWidget(_score_widget("종합",   ov, "#10b981"))
        return frame

    # ─────────────────────────────────────────────────────────────
    #  Tab 1: 비언어 분석
    # ─────────────────────────────────────────────────────────────
    def _tab_nonverbal(self):
        vr    = self.results.get("video", {})
        gaze  = vr.get("gaze", {})
        pose  = vr.get("posture", {})
        hands = vr.get("hands", {})

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(16)
        scroll.setWidget(w)

        # ── 시선 도넛 차트 ──────────────────────────────────────
        lay.addWidget(QLabel("📌 시선 분포", styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._gaze_chart(gaze))

        # ── 자세 지표 ───────────────────────────────────────────
        lay.addWidget(QLabel("🧍 자세 분석", styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._posture_chart(pose))

        # ── 손/다리 떨림 ────────────────────────────────────────
        lay.addWidget(QLabel("✋ 손·다리 분석", styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._hands_legs_chart(hands, pose))

        return scroll

    def _gaze_chart(self, gaze):
        sizes  = [
            gaze.get("camera_ratio", 0.5) * 100,
            gaze.get("down_ratio",   0.2) * 100,
            gaze.get("away_ratio",   0.3) * 100
        ]
        labels = [f"카메라 응시\n{sizes[0]:.0f}%",
                  f"하방 응시\n{sizes[1]:.0f}%",
                  f"회피/측면\n{sizes[2]:.0f}%"]
        colors = ["#6c63ff", "#f43f5e", "#f59e0b"]

        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#13132a")
        ax.set_facecolor("#13132a")
        wedges, _ = ax.pie(
            sizes, labels=None, colors=colors,
            startangle=90, wedgeprops=dict(width=0.55, edgecolor="#0f0f1a"))
        ax.legend(wedges, labels, loc="center right",
                  bbox_to_anchor=(1.6, 0.5), frameon=False,
                  labelcolor="white", fontsize=10)
        ax.set_title("시선 방향 분포", color="#aaa", fontsize=12, pad=10)
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(260)
        plt.close(fig)
        return canvas

    def _posture_chart(self, pose):
        categories = ["어깨 대칭", "척추 정렬", "다리 안정"]
        vals = [
            pose.get("shoulder_symmetry", 0.8) * 10,
            pose.get("spine_alignment",   0.8) * 10,
            10 - pose.get("leg_shake_score", 0)
        ]
        colors = ["#6c63ff" if v >= 7 else "#f59e0b" if v >= 5 else "#f43f5e"
                  for v in vals]
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#13132a")
        ax.set_facecolor("#13132a")
        bars = ax.barh(categories, vals, color=colors, height=0.5)
        ax.set_xlim(0, 10)
        ax.axvline(x=7, color="#444", linestyle="--", linewidth=0.8)
        ax.set_xlabel("점수 (/10)", color="#aaa", fontsize=10)
        ax.tick_params(colors="#aaa")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}", va="center", color="#ddd", fontsize=10)
        plt.setp(ax.spines.values(), color="#2a2a4a")
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(180)
        plt.close(fig)
        return canvas

    def _hands_legs_chart(self, hands, pose):
        fidget   = hands.get("fidget_score", 0)
        leg_shk  = pose.get("leg_shake_score", 0)
        leg_det  = pose.get("leg_shake_detected", False)

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), facecolor="#13132a")
        for ax in axes:
            ax.set_facecolor("#13132a")
            plt.setp(ax.spines.values(), color="#2a2a4a")
            ax.tick_params(colors="#aaa")

        # 손 떨림 게이지
        c1 = "#f43f5e" if fidget > 6 else "#f59e0b" if fidget > 3 else "#10b981"
        axes[0].barh(["손 움직임"], [fidget], color=c1, height=0.4)
        axes[0].barh(["손 움직임"], [10 - fidget], left=[fidget],
                     color="#1a1a30", height=0.4)
        axes[0].set_xlim(0, 10)
        axes[0].set_title("손 꼼지락 지수", color="#aaa", fontsize=11)
        axes[0].text(5, 0, f"{fidget:.1f}/10", ha="center", va="center",
                     color="#fff", fontsize=13, fontweight="bold")

        # 다리 떨음
        c2 = "#f43f5e" if leg_det else "#10b981"
        label2 = f"{'감지됨 ⚠️' if leg_det else '정상 ✅'}"
        axes[1].barh(["다리 떨음"], [leg_shk], color=c2, height=0.4)
        axes[1].barh(["다리 떨음"], [max(0, 10 - leg_shk)], left=[leg_shk],
                     color="#1a1a30", height=0.4)
        axes[1].set_xlim(0, 10)
        axes[1].set_title(f"다리 떨음 {label2}", color="#aaa", fontsize=11)

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(180)
        plt.close(fig)
        return canvas

    # ─────────────────────────────────────────────────────────────
    #  Tab 2: 언어 분석
    # ─────────────────────────────────────────────────────────────
    def _tab_verbal(self):
        ar     = self.results.get("audio", {})
        pitch  = ar.get("pitch", {})
        rate   = ar.get("speech_rate", {})
        filler = ar.get("filler_words", {})
        stutter= ar.get("stuttering", {})

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(16)
        scroll.setWidget(w)

        lay.addWidget(QLabel("📈 피치(음 높낮이) 추이",
                              styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._pitch_chart(pitch))

        lay.addWidget(QLabel("⏱ 말 속도",
                              styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._rate_chart(rate))

        lay.addWidget(QLabel("💬 필러워드 분석",
                              styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;"))
        lay.addWidget(self._filler_chart(filler))

        # 버벅거림 텍스트
        lay.addWidget(QLabel(f"🔁 버벅거림: {stutter.get('count',0)}회",
                              styleSheet="font-size:13px;color:#aaa;"))
        if stutter.get("events"):
            for ev in stutter["events"][:5]:
                lbl = QLabel(f"  ⏱ {ev['time']}s — {ev['text']}")
                lbl.setStyleSheet("color:#f59e0b; font-size:12px;")
                lay.addWidget(lbl)
        return scroll

    def _pitch_chart(self, pitch):
        series = [v for v in pitch.get("pitch_series", []) if v]
        if not series:
            lbl = QLabel("피치 데이터 없음 (발화 감지 불가)")
            lbl.setStyleSheet("color:#555; font-size:13px;")
            return lbl

        fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#13132a")
        ax.set_facecolor("#13132a")
        xs = np.linspace(0, len(series), len(series))
        ax.plot(xs, series, color="#6c63ff", linewidth=1.2, alpha=0.9)
        ax.fill_between(xs, series, alpha=0.15, color="#6c63ff")
        ax.axhline(pitch.get("mean_hz", 0), color="#a855f7",
                   linestyle="--", linewidth=0.8, label=f"평균 {pitch.get('mean_hz',0):.0f}Hz")
        ax.set_ylabel("Hz", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        plt.setp(ax.spines.values(), color="#2a2a4a")
        ax.legend(frameon=False, labelcolor="#aaa", fontsize=9)
        mono = pitch.get("monotony_score", 5)
        ax.set_title(f"피치 변화 추이 | 단조로움 지수 {mono:.1f}/10",
                     color="#aaa", fontsize=11)
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        plt.close(fig)
        return canvas

    def _rate_chart(self, rate):
        wpm     = rate.get("wpm", 0)
        seg_r   = rate.get("segment_rates", [])

        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), facecolor="#13132a")
        for ax in axes:
            ax.set_facecolor("#13132a")
            plt.setp(ax.spines.values(), color="#2a2a4a")
            ax.tick_params(colors="#aaa", labelsize=8)

        # WPM 게이지
        color = "#10b981" if 100 <= wpm <= 180 else "#f43f5e"
        axes[0].barh(["속도"], [wpm], color=color, height=0.4)
        axes[0].axvline(100, color="#10b981", linestyle="--", linewidth=0.8)
        axes[0].axvline(180, color="#f43f5e", linestyle="--", linewidth=0.8)
        axes[0].set_xlim(0, max(250, wpm + 30))
        axes[0].set_title(f"말 속도: {wpm:.0f} WPM", color="#aaa", fontsize=11)
        axes[0].text(wpm + 2, 0, f"{wpm:.0f}", va="center", color="#fff", fontsize=11)

        # 구간별 속도 변동
        if seg_r:
            axes[1].plot(seg_r, "o-", color="#a855f7", markersize=4, linewidth=1.2)
            axes[1].axhline(np.mean(seg_r), color="#6c63ff",
                            linestyle="--", linewidth=0.8)
            axes[1].set_title("구간별 속도 변동 (WPS)", color="#aaa", fontsize=11)
        else:
            axes[1].text(0.5, 0.5, "데이터 없음", transform=axes[1].transAxes,
                         ha="center", va="center", color="#555")

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(190)
        plt.close(fig)
        return canvas

    def _filler_chart(self, filler):
        bd = filler.get("breakdown", {})
        if not bd:
            lbl = QLabel("필러워드 없음 ✅")
            lbl.setStyleSheet("color:#10b981; font-size:13px;")
            return lbl

        words  = list(bd.keys())[:10]
        counts = [bd[w] for w in words]

        fig, ax = plt.subplots(figsize=(8, max(2.5, len(words) * 0.4)),
                               facecolor="#13132a")
        ax.set_facecolor("#13132a")
        plt.setp(ax.spines.values(), color="#2a2a4a")
        ax.tick_params(colors="#aaa", labelsize=9)
        colors = ["#f43f5e" if c > 5 else "#f59e0b" if c > 2 else "#6c63ff"
                  for c in counts]
        bars = ax.barh(words, counts, color=colors, height=0.6)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(c), va="center", color="#ddd", fontsize=10)
        sev = filler.get("severity", "양호")
        ppm = filler.get("per_minute", 0)
        ax.set_title(f"필러워드 — 분당 {ppm:.1f}회 | 심각도: {sev}",
                     color="#aaa", fontsize=11)
        ax.set_xlabel("횟수", color="#aaa", fontsize=9)
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(max(200, len(words) * 30 + 60))
        plt.close(fig)
        return canvas

    # ─────────────────────────────────────────────────────────────
    #  Tab 3: AI 피드백
    # ─────────────────────────────────────────────────────────────
    def _tab_feedback(self):
        fb = self.results.get("feedback", {})

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)
        scroll.setWidget(w)

        # 요약
        summary = fb.get("overall_summary", "")
        if summary:
            summ_lbl = QLabel(f"💬 {summary}")
            summ_lbl.setWordWrap(True)
            summ_lbl.setStyleSheet(
                "background:#1a1a3a;border-left:3px solid #6c63ff;"
                "padding:12px;border-radius:6px;font-size:13px;color:#ddd;")
            lay.addWidget(summ_lbl)

        # 핵심 문제
        issues = fb.get("critical_issues", [])
        if issues:
            lay.addWidget(QLabel("⚠️ 핵심 개선 사항",
                                  styleSheet="font-size:15px;font-weight:600;color:#f43f5e;margin-top:8px;"))
            for iss in issues:
                prio = iss.get("priority", 1)
                color = "#f43f5e" if prio == 1 else "#f59e0b" if prio == 2 else "#6c63ff"
                box = QLabel(f"  [{prio}순위] {iss.get('issue','')}\n  → {iss.get('detail','')}")
                box.setWordWrap(True)
                box.setStyleSheet(
                    f"background:#1a1a30;border-left:3px solid {color};"
                    f"padding:10px;border-radius:6px;font-size:12px;color:#ddd;margin:3px 0;")
                lay.addWidget(box)

        # 비언어 피드백
        nv_fb = fb.get("nonverbal_feedback", {})
        if nv_fb:
            lay.addWidget(QLabel("👁 비언어 피드백",
                                  styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;margin-top:8px;"))
            for key, val in nv_fb.items():
                icons = {"gaze":"👀","posture":"🧍","hands":"✋","legs":"🦵"}
                icon  = icons.get(key, "•")
                self._add_fb_row(lay, f"{icon} {key}", val)

        # 언어 피드백
        vb_fb = fb.get("verbal_feedback", {})
        if vb_fb:
            lay.addWidget(QLabel("🎤 언어 피드백",
                                  styleSheet="font-size:15px;font-weight:600;color:#a0a0c0;margin-top:8px;"))
            icons_v = {"pitch":"🎵","rate":"⏱","filler_words":"💬",
                       "stuttering":"🔁","content":"📝"}
            for key, val in vb_fb.items():
                icon = icons_v.get(key, "•")
                self._add_fb_row(lay, f"{icon} {key}", val)

        # 실천 팁
        tips = fb.get("practice_tips", [])
        if tips:
            lay.addWidget(QLabel("✅ 실천 팁",
                                  styleSheet="font-size:15px;font-weight:600;color:#10b981;margin-top:8px;"))
            for i, tip in enumerate(tips, 1):
                lbl = QLabel(f"  {i}. {tip}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("font-size:13px;color:#a0ffcc;padding:4px 0;")
                lay.addWidget(lbl)

        # 잘하는 점
        pos = fb.get("positive_points", [])
        if pos:
            lay.addWidget(QLabel("⭐ 잘하고 있는 점",
                                  styleSheet="font-size:15px;font-weight:600;color:#f59e0b;margin-top:8px;"))
            for p in pos:
                lbl = QLabel(f"  ★ {p}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("font-size:13px;color:#fde68a;padding:4px 0;")
                lay.addWidget(lbl)

        lay.addStretch()
        return scroll

    def _add_fb_row(self, lay, key, val):
        box = QLabel(f"  <b>{key}</b><br>{val}")
        box.setWordWrap(True)
        box.setTextFormat(Qt.RichText)
        box.setStyleSheet(
            "background:#1a1a30;padding:10px;border-radius:6px;"
            "font-size:12px;color:#ddd;margin:2px 0;")
        lay.addWidget(box)

    # ─────────────────────────────────────────────────────────────
    #  Tab 4: 전사 내용
    # ─────────────────────────────────────────────────────────────
    def _tab_transcript(self):
        ar = self.results.get("audio", {})
        text = ar.get("transcript", "(전사 데이터 없음)")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)
        scroll.setWidget(w)

        # 텍스트 박스
        te = QTextEdit()
        te.setPlainText(text)
        te.setReadOnly(True)
        te.setStyleSheet(
            "background:#13132a;color:#ddd;font-size:14px;"
            "border:1px solid #2a2a4a;border-radius:8px;padding:12px;"
            "line-height:1.6;")
        te.setMinimumHeight(300)
        lay.addWidget(te)

        # 세그먼트 타임라인
        segments = ar.get("word_segments", [])
        if segments:
            lay.addWidget(QLabel("⏱ 발화 타임라인",
                                  styleSheet="font-size:14px;font-weight:600;color:#a0a0c0;margin-top:12px;"))
            for seg in segments[:30]:
                t   = seg.get("start", 0)
                txt = seg.get("text", "").strip()
                row = QLabel(f"  [{t:>6.1f}s]  {txt}")
                row.setStyleSheet("font-size:11px;color:#888;font-family:monospace;")
                lay.addWidget(row)

        lay.addStretch()
        return scroll

    # ─────────────────────────────────────────────────────────────
    def _save_results(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "speech_analysis.json",
            "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2,
                          default=str)
            QMessageBox.information(self, "저장 완료", f"저장 완료:\n{path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename="crash.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    import sys as _sys
    def _excepthook(exc_type, exc_val, exc_tb):
        import traceback as _tb
        msg = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
        logging.critical(msg)
        print(msg)
    _sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())