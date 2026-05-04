"""
audio_analyzer.py
librosa + Whisper 기반 음성 분석 모듈
- 피치, 속도, 필러워드, 버벅거림, 감정 톤 분석
"""

import os
import re
import subprocess
import tempfile
import warnings
import numpy as np
import librosa
from faster_whisper import WhisperModel
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
#  한국어/영어 필러워드 목록 (민감도 핵심)
# ──────────────────────────────────────────────────────────────────────────────
FILLER_KO = [
    "어", "음", "아", "그", "저", "뭐", "이제", "근데", "그런데",
    "사실", "솔직히", "뭔가", "어떻게 보면", "좀", "막", "그냥", "약간",
    "이렇게", "저렇게", "그렇게", "거", "게", "요", "이거", "저거",
    "있잖아", "있잖아요", "그니까", "그러니까"
]
FILLER_EN = [
    "um", "uh", "uhh", "umm", "like", "you know", "so", "right",
    "basically", "literally", "actually", "honestly", "kind of",
    "sort of", "i mean", "you see", "well"
]

STUTTER_PATTERN = re.compile(
    r'\b(\w{1,3})-\1|\b(\w+)\s+\2\b', re.IGNORECASE | re.UNICODE)


class AudioAnalyzer:
    def __init__(self, whisper_model: str = "base"):
        self.whisper_model_name = whisper_model
        self._whisper = None

    def _get_whisper(self):
        if self._whisper is None:
            self._whisper = WhisperModel(
                self.whisper_model_name,
                device="cpu",
                compute_type="int8"
            )
        return self._whisper

    # ─────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────
    def analyze(self, video_path: str, progress_callback=None) -> dict:
        import logging
        log = logging.getLogger("worker")

        # 1. 오디오 추출
        log.info("AUDIO-1: 오디오 추출 시작")
        audio_path = self._extract_audio(video_path)
        log.info(f"AUDIO-1 완료: {audio_path}")
        if progress_callback:
            progress_callback(62)

        # 2. librosa 로드
        log.info("AUDIO-2: librosa 로드 시작")
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        log.info(f"AUDIO-2 완료: 길이={len(y)/sr:.1f}초")

        # 3. Whisper 전사
        if progress_callback:
            progress_callback(65)
        log.info("AUDIO-3: Whisper 전사 시작")
        transcript_data = self._transcribe(audio_path)
        log.info(f"AUDIO-3 완료: {len(transcript_data.get('text',''))}자")

        if progress_callback:
            progress_callback(80)

        # 4. 각종 분석
        log.info("AUDIO-4: 피치 분석")
        pitch_r   = self._analyze_pitch(y, sr)
        log.info("AUDIO-5: 속도 분석")
        rate_r    = self._analyze_speech_rate(transcript_data, y, sr)
        log.info("AUDIO-6: 필러워드 분석")
        filler_r  = self._detect_fillers(transcript_data)
        log.info("AUDIO-7: 버벅거림 분석")
        stutter_r = self._detect_stuttering(transcript_data)
        log.info("AUDIO-8: 침묵 분석")
        pause_r   = self._analyze_pauses(y, sr)
        log.info("AUDIO-9: 에너지 분석")
        energy_r  = self._analyze_energy(y, sr)
        log.info("AUDIO-10: 볼륨 분석")
        volume_r  = self._analyze_volume_variation(y, sr)

        # 5. 임시 파일 정리
        try:
            os.remove(audio_path)
        except Exception:
            pass

        if progress_callback:
            progress_callback(90)

        log.info("AUDIO 분석 완료")
        return {
            "transcript":     transcript_data.get("text", ""),
            "word_segments":  transcript_data.get("segments", []),
            "pitch":          pitch_r,
            "speech_rate":    rate_r,
            "filler_words":   filler_r,
            "stuttering":     stutter_r,
            "pauses":         pause_r,
            "energy":         energy_r,
            "volume":         volume_r,
            "overall_verbal_score": self._verbal_score(
                pitch_r, rate_r, filler_r, stutter_r, pause_r)
        }

    # ─────────────────────────────────────────────────────────────
    #  오디오 추출
    # ─────────────────────────────────────────────────────────────
    def _extract_audio(self, video_path: str) -> str:
        import tempfile, subprocess
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", video_path,
                 "-ac", "1", "-ar", "16000", "-vn", out],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True)
        except Exception:
            # 최후 폴백: librosa 직접 로드
            y, sr = librosa.load(video_path, sr=16000, mono=True)
            import soundfile as sf
            sf.write(out, y, sr)
        return out

    # ─────────────────────────────────────────────────────────────
    #  Whisper 전사
    # ─────────────────────────────────────────────────────────────
    def _transcribe(self, audio_path: str) -> dict:
        model = self._get_whisper()
        segments_gen, info = model.transcribe(
            audio_path,
            word_timestamps=True
        )
        segments = []
        full_text = ""
        for seg in segments_gen:
            text = seg.text.strip()
            full_text += text + " "
            segments.append({
                "start": seg.start,
                "end":   seg.end,
                "text":  text
            })
        return {"text": full_text.strip(), "segments": segments}

    # ─────────────────────────────────────────────────────────────
    #  피치 분석
    # ─────────────────────────────────────────────────────────────
    def _analyze_pitch(self, y, sr) -> dict:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr)

        voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced) < 5:
            return {"mean_hz": 0, "std_hz": 0, "monotony_score": 5.0,
                    "variation_ok": True, "pitch_series": []}

        mean_hz = float(np.mean(voiced))
        std_hz  = float(np.std(voiced))
        cv      = std_hz / (mean_hz + 1e-6)   # 변동계수

        # 단조로움 점수: cv 낮을수록 단조 (높을수록 표현력 풍부)
        monotony = max(0.0, 10.0 - cv * 60)   # cv≈0.17 → 점수 0

        # 피치 시계열 (시각화용, 최대 200 포인트)
        n_pts = min(200, len(f0))
        step  = max(1, len(f0) // n_pts)
        series = [
            round(float(v), 1) if (not np.isnan(v) and v > 0) else None
            for v in f0[::step][:n_pts]
        ]

        return {
            "mean_hz":      round(mean_hz, 1),
            "std_hz":       round(std_hz, 1),
            "monotony_score": round(monotony, 1),  # 높을수록 단조
            "variation_ok": cv > 0.08,
            "pitch_series": series
        }

    # ─────────────────────────────────────────────────────────────
    #  말 속도
    # ─────────────────────────────────────────────────────────────
    def _analyze_speech_rate(self, transcript_data, y, sr) -> dict:
        text     = transcript_data.get("text", "")
        segments = transcript_data.get("segments", [])

        total_dur = librosa.get_duration(y=y, sr=sr)

        # 발화 구간에서 단어 수 세기
        words = text.split()
        word_count = len(words)

        # 한국어는 글자 수 기반이 더 정확
        char_count = len(re.sub(r'\s', '', text))

        # 발화 구간만의 실제 시간 (무음 제외)
        speech_dur = 0.0
        for seg in segments:
            speech_dur += seg.get("end", 0) - seg.get("start", 0)
        speech_dur = max(speech_dur, 1.0)

        wpm  = word_count / (speech_dur / 60)
        cpm  = char_count / (speech_dur / 60)  # 한국어 분당 글자

        # 속도 변동: 구간별 WPS 계산
        seg_rates = []
        for seg in segments:
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur > 0.5:
                wc = len(seg.get("text", "").split())
                seg_rates.append(wc / dur)

        rate_std = float(np.std(seg_rates)) if len(seg_rates) > 2 else 0.0

        # 권장 범위: 한국어 면접 130~180 WPM
        ideal_min, ideal_max = 100, 180
        too_fast  = wpm > ideal_max
        too_slow  = wpm < ideal_min
        rate_ok   = ideal_min <= wpm <= ideal_max

        return {
            "wpm":            round(wpm, 1),
            "cpm":            round(cpm, 1),
            "speech_duration":round(speech_dur, 1),
            "total_duration": round(total_dur, 1),
            "rate_variation": round(rate_std, 2),
            "too_fast":       too_fast,
            "too_slow":       too_slow,
            "rate_ok":        rate_ok,
            "segment_rates":  [round(r, 2) for r in seg_rates]
        }

    # ─────────────────────────────────────────────────────────────
    #  필러워드 (고민감도 핵심 기능)
    # ─────────────────────────────────────────────────────────────
    def _detect_fillers(self, transcript_data) -> dict:
        text     = transcript_data.get("text", "").lower()
        segments = transcript_data.get("segments", [])

        breakdown = {}
        events    = []

        all_fillers = FILLER_KO + FILLER_EN

        for filler in all_fillers:
            pattern = re.compile(
                r'(?<!\w)' + re.escape(filler) + r'(?!\w)',
                re.IGNORECASE)
            matches = list(pattern.finditer(text))
            if matches:
                breakdown[filler] = len(matches)

        # 세그먼트별 타임스탬프
        for seg in segments:
            seg_text = seg.get("text", "").lower()
            for filler in all_fillers:
                pattern = re.compile(
                    r'(?<!\w)' + re.escape(filler) + r'(?!\w)',
                    re.IGNORECASE)
                if pattern.search(seg_text):
                    events.append({
                        "time":  round(seg.get("start", 0), 1),
                        "word":  filler,
                        "text":  seg.get("text", "").strip()
                    })

        total = sum(breakdown.values())
        dur   = max(1.0, sum(
            seg.get("end", 0) - seg.get("start", 0) for seg in segments))
        per_min = total / (dur / 60)

        # 심각도: 분당 필러워드 수 기준
        severity = "심각" if per_min > 10 else \
                   "주의" if per_min > 5  else \
                   "양호" if per_min > 2  else "좋음"

        return {
            "total":     total,
            "per_minute":round(per_min, 1),
            "breakdown": dict(sorted(breakdown.items(),
                                     key=lambda x: x[1], reverse=True)),
            "events":    events[:30],
            "severity":  severity
        }

    # ─────────────────────────────────────────────────────────────
    #  버벅거림 / 반복
    # ─────────────────────────────────────────────────────────────
    def _detect_stuttering(self, transcript_data) -> dict:
        segments = transcript_data.get("segments", [])
        events   = []
        count    = 0

        for seg in segments:
            text = seg.get("text", "")
            matches = STUTTER_PATTERN.findall(text)
            if matches:
                count += len(matches)
                events.append({
                    "time": round(seg.get("start", 0), 1),
                    "text": text.strip()
                })

            # 연속 짧은 세그먼트 = 버벅임
            words = text.split()
            for i in range(len(words) - 1):
                if (words[i].lower() == words[i+1].lower() and
                        len(words[i]) > 1):
                    count += 1
                    events.append({
                        "time": round(seg.get("start", 0), 1),
                        "text": text.strip()
                    })

        return {"count": count, "events": events[:15]}

    # ─────────────────────────────────────────────────────────────
    #  침묵 / 포즈
    # ─────────────────────────────────────────────────────────────
    def _analyze_pauses(self, y, sr) -> dict:
        intervals = librosa.effects.split(y, top_db=30)
        pauses = []
        prev_end = 0
        for start, end in intervals:
            if start > prev_end:
                dur = (start - prev_end) / sr
                if dur > 0.5:
                    pauses.append({"start": round(prev_end/sr, 1),
                                   "duration": round(dur, 2)})
            prev_end = end

        long_pauses = [p for p in pauses if p["duration"] > 2.0]
        return {
            "pause_count":      len(pauses),
            "long_pause_count": len(long_pauses),
            "avg_duration":     round(float(np.mean([p["duration"]
                                  for p in pauses])), 2) if pauses else 0.0,
            "max_duration":     round(float(max((p["duration"]
                                  for p in pauses), default=0)), 2),
            "events":           long_pauses[:10]
        }

    # ─────────────────────────────────────────────────────────────
    #  에너지 / 볼륨
    # ─────────────────────────────────────────────────────────────
    def _analyze_energy(self, y, sr) -> dict:
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        return {
            "mean_db":  round(float(np.mean(rms_db)), 1),
            "std_db":   round(float(np.std(rms_db)), 1),
            "variation_ok": float(np.std(rms_db)) > 5.0
        }

    def _analyze_volume_variation(self, y, sr) -> dict:
        chunk = sr * 5
        volumes = []
        for i in range(0, len(y), chunk):
            seg = y[i:i+chunk]
            if len(seg) > 100:
                rms = float(np.sqrt(np.mean(seg**2)))
                volumes.append(round(rms, 5))
        return {
            "per_segment": volumes,
            "cv": round(float(np.std(volumes) / (np.mean(volumes) + 1e-9)), 3)
                  if volumes else 0.0
        }

    # ─────────────────────────────────────────────────────────────
    #  종합 언어 점수
    # ─────────────────────────────────────────────────────────────
    def _verbal_score(self, pitch, rate, filler, stutter, pause) -> float:
        s = 10.0
        # 필러워드
        fpm = filler.get("per_minute", 0)
        s -= min(3.0, fpm * 0.3)
        # 속도
        if rate.get("too_fast") or rate.get("too_slow"):
            s -= 1.5
        # 단조로움
        mono = pitch.get("monotony_score", 5)
        s -= mono * 0.15
        # 버벅거림
        s -= min(2.0, stutter.get("count", 0) * 0.3)
        # 긴 침묵
        s -= min(1.5, pause.get("long_pause_count", 0) * 0.3)
        return round(max(0.0, s), 1)