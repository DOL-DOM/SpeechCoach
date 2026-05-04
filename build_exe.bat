@echo off
REM ============================================================
REM  SpeechCoach AI — Windows EXE 빌드 스크립트
REM  실행 전 requirements.txt 설치 완료 필요
REM ============================================================

echo [1/3] PyInstaller 설치 확인...
pip install pyinstaller --quiet

echo [2/3] EXE 빌드 중... (수 분 소요)

pyinstaller ^
  --name "SpeechCoach" ^
  --onedir ^
  --windowed ^
  --icon=icon.ico ^
  --add-data "video_analyzer.py;." ^
  --add-data "audio_analyzer.py;." ^
  --add-data "feedback_generator.py;." ^
  --hidden-import mediapipe ^
  --hidden-import mediapipe.python.solutions.face_mesh ^
  --hidden-import mediapipe.python.solutions.pose ^
  --hidden-import mediapipe.python.solutions.hands ^
  --hidden-import whisper ^
  --hidden-import librosa ^
  --hidden-import anthropic ^
  --hidden-import PyQt5 ^
  --hidden-import matplotlib ^
  --collect-all mediapipe ^
  --collect-all whisper ^
  --collect-all librosa ^
  --noconfirm ^
  main.py

echo [3/3] 완료!
echo.
echo 빌드 결과: dist\SpeechCoach\SpeechCoach.exe
echo.
echo ★ 참고: Whisper 모델은 첫 실행 시 자동 다운로드됩니다 (~150MB for 'base')
echo ★ ffmpeg가 설치되어 있으면 영상 처리 속도가 빨라집니다
echo.
pause
