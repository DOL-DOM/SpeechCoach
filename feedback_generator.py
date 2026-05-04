"""
feedback_generator.py
Anthropic Claude API를 사용한 맞춤형 교정 피드백 생성
"""

import json
from anthropic import Anthropic


class FeedbackGenerator:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate(self, video_result: dict, audio_result: dict,
                 context: str = "면접") -> dict:
        """
        분석 결과를 바탕으로 Claude API를 통해 구조화된 피드백 생성
        """
        prompt = self._build_prompt(video_result, audio_result, context)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text
        return self._parse_feedback(raw)

    # ─────────────────────────────────────────────────────────────
    def _build_prompt(self, vr: dict, ar: dict, context: str) -> str:
        gaze    = vr.get("gaze", {})
        posture = vr.get("posture", {})
        hands   = vr.get("hands", {})
        pitch   = ar.get("pitch", {})
        rate    = ar.get("speech_rate", {})
        filler  = ar.get("filler_words", {})
        stutter = ar.get("stuttering", {})
        pause   = ar.get("pauses", {})
        energy  = ar.get("energy", {})
        transcript = ar.get("transcript", "")[:600]  # 최대 600자

        nv_score = vr.get("overall_nonverbal_score", 5)
        v_score  = ar.get("overall_verbal_score", 5)

        data_summary = f"""
=== 비디오 분석 결과 ===
- 시선: 카메라 응시 {gaze.get('camera_ratio',0)*100:.0f}%, 하방 응시 {gaze.get('down_ratio',0)*100:.0f}%, 다른 곳 {gaze.get('away_ratio',0)*100:.0f}%
- 평균 고개 각도(pitch): {gaze.get('avg_head_pitch',0):.1f}° (음수=숙임)
- 자세 점수: {posture.get('posture_score',5)}/10
- 어깨 대칭: {posture.get('shoulder_symmetry',0.8)*100:.0f}%
- 척추 정렬: {posture.get('spine_alignment',0.8)*100:.0f}%
- 다리 떨음: {'감지됨' if posture.get('leg_shake_detected') else '없음'} (강도:{posture.get('leg_shake_score',0):.1f})
- 손 떨림/꼼지락: {hands.get('fidget_score',0):.1f}/10 (높을수록 심각)
- 비언어 종합: {nv_score}/10

=== 음성 분석 결과 ===
- 평균 피치: {pitch.get('mean_hz',0):.0f}Hz
- 피치 변동: {pitch.get('std_hz',0):.1f}Hz (단조로움 {pitch.get('monotony_score',5):.1f}/10)
- 말 속도: {rate.get('wpm',0):.0f} WPM ({'너무 빠름' if rate.get('too_fast') else '너무 느림' if rate.get('too_slow') else '적절'})
- 속도 변동: {rate.get('rate_variation',0):.2f}
- 필러워드: 총 {filler.get('total',0)}회 (분당 {filler.get('per_minute',0):.1f}회) - {filler.get('severity','양호')}
- 자주 쓰는 필러: {', '.join([f"{k}({v}회)" for k,v in list(filler.get('breakdown',{}).items())[:5]])}
- 버벅거림: {stutter.get('count',0)}회
- 긴 침묵: {pause.get('long_pause_count',0)}회 (평균 {pause.get('avg_duration',0):.1f}초)
- 언어 종합: {v_score}/10

=== 발화 내용 (일부) ===
{transcript}
"""

        return f"""당신은 '{context}' 스피치 전문 코치입니다.
아래 AI 분석 데이터를 바탕으로 발표자에게 구체적이고 실행 가능한 피드백을 제공하세요.

{data_summary}

다음 형식으로 피드백을 작성하세요 (JSON):
{{
  "overall_summary": "2-3문장 전반적 평가",
  "scores": {{
    "nonverbal": {nv_score},
    "verbal": {v_score},
    "overall": <평균 점수 계산>
  }},
  "critical_issues": [
    {{"issue": "가장 중요한 문제", "detail": "구체적 설명", "priority": 1}},
    {{"issue": "두 번째 문제", "detail": "구체적 설명", "priority": 2}},
    {{"issue": "세 번째 문제", "detail": "구체적 설명", "priority": 3}}
  ],
  "nonverbal_feedback": {{
    "gaze": "시선처리 피드백 (구체적 개선 방법 포함)",
    "posture": "자세 피드백",
    "hands": "손동작 피드백",
    "legs": "다리/하체 피드백"
  }},
  "verbal_feedback": {{
    "pitch": "목소리 높낮이 피드백",
    "rate": "말 속도 피드백",
    "filler_words": "필러워드 개선 방법 (대체 표현 제시)",
    "stuttering": "버벅거림 개선 방법",
    "content": "발화 내용/표현 피드백"
  }},
  "practice_tips": [
    "실천 팁 1",
    "실천 팁 2",
    "실천 팁 3"
  ],
  "positive_points": [
    "잘 하고 있는 점 1",
    "잘 하고 있는 점 2"
  ]
}}

반드시 JSON만 출력하고 다른 텍스트는 포함하지 마세요."""

    # ─────────────────────────────────────────────────────────────
    def _parse_feedback(self, raw: str) -> dict:
        # JSON 블록 추출
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                l for l in lines if not l.startswith("```")).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본 구조 반환
            return {
                "overall_summary": raw[:300],
                "scores": {"nonverbal": 5, "verbal": 5, "overall": 5},
                "critical_issues": [],
                "nonverbal_feedback": {},
                "verbal_feedback": {},
                "practice_tips": [],
                "positive_points": []
            }
