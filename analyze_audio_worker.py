"""
analyze_audio_worker.py
별도 프로세스로 실행되는 오디오 분석 스크립트
결과를 JSON으로 stdout에 출력
"""
import sys
import json
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    video_path    = sys.argv[1]
    whisper_model = sys.argv[2] if len(sys.argv) > 2 else "base"
    result_path   = sys.argv[3] if len(sys.argv) > 3 else "audio_result.json"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from audio_analyzer import AudioAnalyzer
    aa = AudioAnalyzer(whisper_model)
    result = aa.analyze(video_path)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=str)
    print(f"저장 완료: {result_path}")

if __name__ == "__main__":
    main()