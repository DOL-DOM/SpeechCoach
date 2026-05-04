# SpeechCoach AI

스피치·면접 영상을 분석해 비언어(시선, 자세, 손동작, 다리 떨음)와 언어(피치, 속도, 필러워드, 버벅거림) 습관을 시각화하고 교정 피드백을 제공하는 로컬 실행 프로그램입니다.

---

## 요구사항

- OS: Windows 10/11 (64-bit)
- Python: 3.11 권장 (3.12 이상은 PyTorch/MediaPipe 호환 문제 있음)
- 인터넷: 최초 실행 시 Whisper 모델 다운로드 필요 (~150MB)

---

## 설치

### 1. Python 3.11 설치

https://www.python.org/downloads/release/python-3119/

페이지 하단에서 **Windows installer (64-bit)** 다운로드 후 설치.
설치 시 **Add Python to PATH** 반드시 체크.

설치 확인:
```
py -3.11 --version
```

### 2. 프로젝트 폴더로 이동

```
cd 프로젝트경로
```

### 3. 가상환경 생성 및 활성화

```
py -3.11 -m venv venv
venv\Scripts\activate
```

터미널 앞에 (venv) 표시가 생기면 활성화된 것입니다.

### 4. 패키지 설치

순서를 지켜서 설치합니다.

```
pip install PyQt5 opencv-python mediapipe==0.10.14 matplotlib numpy scipy Pillow imageio[ffmpeg] soundfile anthropic
```

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```
pip install faster-whisper librosa
```

### 5. Visual C++ 재배포 패키지 설치

PyTorch DLL 로딩에 필요합니다. 아래 링크에서 설치 후 PC를 재시작하세요.

https://aka.ms/vs/17/release/vc_redist.x64.exe

### 6. 실행

```
python main.py
```

---

## 파일 구조

```
프로젝트폴더/
├── main.py                  # GUI 진입점 (PyQt5)
├── video_analyzer.py        # MediaPipe 기반 비언어 분석
├── audio_analyzer.py        # Whisper + librosa 음성 분석
├── analyze_audio_worker.py  # 오디오 분석 subprocess 워커
├── feedback_generator.py    # Claude API 피드백 생성 (선택)
└── README.md
```

---

## 사용 방법

1. 프로그램 실행 후 영상 파일 선택 (.mp4, .mov, .avi, .mkv)
2. Whisper 모델 크기 선택 (기본값 base 권장)
3. 분석 맥락 선택 (면접 / 발표 / 토론 등)
4. 분석 시작 클릭
5. 결과 탭에서 비언어 분석, 언어 분석, 전사 내용 확인

분석 시간은 영상 1분당 약 1~2분 소요됩니다 (CPU 기준).

---

## 분석 항목

**비언어**

| 항목 | 기술 |
|------|------|
| 시선 방향 | MediaPipe Face Mesh iris landmark |
| 고개 각도 | 3D head pose estimation |
| 손 움직임 | Hand landmark 이동량 분산 |
| 다리 떨음 | Pose ankle landmark 변동 감지 |
| 자세 대칭 | 어깨·골반 좌표 비교 |

**언어**

| 항목 | 기술 |
|------|------|
| 음성 전사 | faster-whisper |
| 피치 분석 | librosa pyin |
| 말 속도 | WPM 계산 (권장 범위: 100~180 WPM) |
| 필러워드 | 한국어·영어 30+ 패턴 매칭 |
| 버벅거림 | 반복 단어 패턴 감지 |

---

## 자주 발생하는 오류

**ModuleNotFoundError: No module named 'PyQt5'**
```
pip install PyQt5
```

**OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다**

Visual C++ 재배포 패키지가 없거나 Python 버전이 3.12 이상입니다.
Python 3.11로 재설치 후 vc_redist.x64.exe를 설치하세요.

**AttributeError: module 'mediapipe' has no attribute 'solutions'**
```
pip install mediapipe==0.10.14
```

**프로그램이 분석 중 갑자기 종료됨**

오디오 분석은 별도 프로세스(analyze_audio_worker.py)로 실행됩니다.
해당 파일이 프로젝트 폴더에 있는지 확인하세요.

---

## 참고

- Whisper 모델은 최초 실행 시 HuggingFace에서 자동 다운로드됩니다 (base 기준 약 150MB).
- 전신이 프레임에 들어와야 다리 떨음 감지가 가능합니다.
- AI 피드백 기능(Claude API)은 현재 비활성화 상태입니다. feedback_generator.py와 Anthropic API Key가 있으면 활성화할 수 있습니다.