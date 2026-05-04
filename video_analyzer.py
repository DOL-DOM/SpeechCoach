"""
video_analyzer.py
MediaPipe 기반 비디오 분석 모듈
- 시선(아이트래킹), 자세, 손 움직임, 다리 떨음 등 분석
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")


class VideoAnalyzer:
    # MediaPipe Pose landmark indices
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP       = 23
    RIGHT_HIP      = 24
    LEFT_KNEE      = 25
    RIGHT_KNEE     = 26
    LEFT_ANKLE     = 27
    RIGHT_ANKLE    = 28

    # Face Mesh iris / eye landmarks
    LEFT_IRIS   = [468, 469, 470, 471, 472]
    RIGHT_IRIS  = [473, 474, 475, 476, 477]
    LEFT_EYE_CORNERS  = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose      = mp.solutions.pose
        self.mp_hands     = mp.solutions.hands

    # ─────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────
    def analyze(self, video_path: str, progress_callback=None) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps

        face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose  = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        gaze_log  = []      # [{time, gaze, head_pitch, head_yaw}]
        pose_log  = []      # [{time, landmarks}]
        hand_log  = []      # [{time, pos}]
        blink_log = []      # [{time}]

        sample_every = max(1, int(fps / 5))  # ~5fps sampling
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                ts  = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # ── Face / gaze ──────────────────────────────────
                fm_res = face_mesh.process(rgb)
                if fm_res.multi_face_landmarks:
                    lmks = fm_res.multi_face_landmarks[0]
                    gaze      = self._estimate_gaze(lmks, w, h)
                    head_pose = self._estimate_head_pose(lmks, w, h)
                    ear       = self._eye_aspect_ratio(lmks)
                    gaze_log.append({
                        "time": round(ts, 2),
                        "gaze": gaze,
                        "head_pitch": head_pose["pitch"],
                        "head_yaw":   head_pose["yaw"],
                        "ear": ear
                    })
                    if ear < 0.15:
                        blink_log.append(ts)

                # ── Body pose ─────────────────────────────────────
                pose_res = pose.process(rgb)
                if pose_res.pose_landmarks:
                    pose_log.append({
                        "time": round(ts, 2),
                        "landmarks": [
                            (lm.x, lm.y, lm.z, lm.visibility)
                            for lm in pose_res.pose_landmarks.landmark
                        ]
                    })

                # ── Hands ────────────────────────────────────────
                hand_res = hands.process(rgb)
                if hand_res.multi_hand_landmarks:
                    for hlm in hand_res.multi_hand_landmarks:
                        center = np.mean(
                            [(lm.x, lm.y) for lm in hlm.landmark], axis=0)
                        hand_log.append({"time": round(ts, 2),
                                         "pos": center.tolist()})

                if progress_callback:
                    pct = int(frame_idx / max(total_frames, 1) * 55)
                    progress_callback(pct)

            frame_idx += 1

        cap.release()
        face_mesh.close()
        pose.close()
        hands.close()

        return self._compile(gaze_log, pose_log, hand_log, blink_log, duration)

    # ─────────────────────────────────────────────────────────────
    #  Gaze & head
    # ─────────────────────────────────────────────────────────────
    def _estimate_gaze(self, landmarks, w, h):
        """
        iris 위치 비율 + 얼굴 방향으로 시선 방향 분류:
        'camera' | 'down' | 'left' | 'right' | 'up'
        """
        lm = landmarks.landmark

        def iris_h_ratio(iris_ids, left_corner_id, right_corner_id):
            iris_x = np.mean([lm[i].x for i in iris_ids])
            ex_l = lm[left_corner_id].x
            ex_r = lm[right_corner_id].x
            span = ex_r - ex_l
            if abs(span) < 1e-6:
                return 0.5
            return (iris_x - ex_l) / span

        l_ratio = iris_h_ratio(self.LEFT_IRIS,  33, 133)
        r_ratio = iris_h_ratio(self.RIGHT_IRIS, 362, 263)
        h_ratio  = (l_ratio + r_ratio) / 2          # 0=극좌, 1=극우

        # 수직 시선: nose tip vs eye mid
        nose_y     = lm[1].y
        eye_mid_y  = (lm[33].y + lm[263].y) / 2
        chin_y     = lm[152].y
        face_h     = chin_y - eye_mid_y + 1e-6
        down_ratio = (nose_y - eye_mid_y) / face_h

        if down_ratio > 0.30:   # 아래를 보고 있음
            return "down"
        if nose_y < eye_mid_y - 0.04:
            return "up"
        if h_ratio < 0.38:
            return "right"      # 시청자 기준 왼쪽 → 카메라 기준 오른쪽
        if h_ratio > 0.62:
            return "left"
        return "camera"

    def _estimate_head_pose(self, landmarks, w, h):
        """간단한 pitch / yaw 추정"""
        lm = landmarks.landmark
        nose  = np.array([lm[1].x,   lm[1].y,   lm[1].z])
        chin  = np.array([lm[152].x, lm[152].y, lm[152].z])
        l_eye = np.array([lm[33].x,  lm[33].y,  lm[33].z])
        r_eye = np.array([lm[263].x, lm[263].y, lm[263].z])

        face_vec = chin - nose
        pitch = float(np.degrees(
            np.arctan2(-face_vec[1],
                       np.sqrt(face_vec[0]**2 + face_vec[2]**2))))

        eye_vec = r_eye - l_eye
        yaw = float(np.degrees(
            np.arctan2(eye_vec[2], abs(eye_vec[0]) + 1e-6)))

        return {"pitch": round(pitch, 1), "yaw": round(yaw, 1)}

    def _eye_aspect_ratio(self, landmarks):
        """눈 가로/세로 비율 → 눈 감음(blink) 감지"""
        lm = landmarks.landmark
        # 수직 거리 (위/아래 눈꺼풀)
        v1 = abs(lm[159].y - lm[145].y)
        v2 = abs(lm[386].y - lm[374].y)
        # 수평 거리
        h1 = abs(lm[33].x  - lm[133].x)
        h2 = abs(lm[362].x - lm[263].x)
        ear = (v1 + v2) / (h1 + h2 + 1e-6)
        return round(ear, 3)

    # ─────────────────────────────────────────────────────────────
    #  Posture
    # ─────────────────────────────────────────────────────────────
    def _analyze_posture(self, pose_log):
        if not pose_log:
            return {
                "posture_score": 5.0,
                "shoulder_symmetry": 0.8,
                "spine_alignment": 0.8,
                "leg_shake_detected": False,
                "leg_shake_score": 0.0
            }

        sym_scores   = []
        spine_scores = []
        ankle_ys     = []

        LS, RS = self.LEFT_SHOULDER, self.RIGHT_SHOULDER
        LH, RH = self.LEFT_HIP,      self.RIGHT_HIP
        LA, RA = self.LEFT_ANKLE,    self.RIGHT_ANKLE

        for fd in pose_log:
            lms = fd["landmarks"]
            if len(lms) <= RH:
                continue

            ls = lms[LS]; rs = lms[RS]
            lh = lms[LH]; rh = lms[RH]

            # 어깨 대칭
            if ls[3] > 0.5 and rs[3] > 0.5:
                sym = 1.0 - min(1.0, abs(ls[1] - rs[1]) * 8)
                sym_scores.append(sym)

            # 척추 정렬 (어깨/골반 중심 정렬)
            vis = [lms[i][3] > 0.5 for i in [LS, RS, LH, RH]]
            if all(vis):
                sh_mid = (ls[0] + rs[0]) / 2
                hp_mid = (lh[0] + rh[0]) / 2
                spine_scores.append(1.0 - min(1.0, abs(sh_mid - hp_mid) * 6))

            # 발목 Y 추적 (다리 떨음)
            if len(lms) > RA:
                la_vis = lms[LA][3]; ra_vis = lms[RA][3]
                if la_vis > 0.3:
                    ankle_ys.append(lms[LA][1])
                elif ra_vis > 0.3:
                    ankle_ys.append(lms[RA][1])

        # 다리 떨음 판별 (분산 기준)
        leg_shake = False
        leg_score = 0.0
        if len(ankle_ys) >= 10:
            var = float(np.var(ankle_ys))
            # 연속 고주파 변동 감지
            diffs = np.abs(np.diff(ankle_ys))
            high_freq = np.mean(diffs > np.mean(diffs) + np.std(diffs))
            leg_score = min(10.0, var * 2000 + high_freq * 5)
            leg_shake = var > 0.0003 or high_freq > 0.3

        sym  = float(np.mean(sym_scores))   if sym_scores   else 0.8
        spal = float(np.mean(spine_scores)) if spine_scores else 0.8
        score = sym * 4 + spal * 4 + (2 if not leg_shake else 0)

        return {
            "posture_score":      round(score, 1),
            "shoulder_symmetry":  round(sym, 2),
            "spine_alignment":    round(spal, 2),
            "leg_shake_detected": leg_shake,
            "leg_shake_score":    round(leg_score, 1)
        }

    # ─────────────────────────────────────────────────────────────
    #  Hands
    # ─────────────────────────────────────────────────────────────
    def _analyze_hands(self, hand_log):
        if len(hand_log) < 3:
            return {"fidget_score": 0.0, "events": [],
                    "movement_variance": 0.0}

        positions = np.array([h["pos"] for h in hand_log])
        times     = [h["time"] for h in hand_log]
        movements = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        mean_m = float(np.mean(movements))
        std_m  = float(np.std(movements))
        thr    = mean_m + 1.5 * std_m

        fidget_frames = movements > thr
        fidget_ratio  = float(np.mean(fidget_frames))
        fidget_score  = min(10.0, fidget_ratio * 10)

        # 이벤트 병합
        events = []
        in_f = False; t_start = 0
        for i, (f, t) in enumerate(zip(fidget_frames, times[1:])):
            if f and not in_f:
                in_f = True; t_start = t
            elif not f and in_f:
                in_f = False
                dur = t - t_start
                if dur > 0.4:
                    events.append({"time": round(t_start, 1),
                                   "duration": round(dur, 1)})
        return {
            "fidget_score":      round(fidget_score, 1),
            "movement_variance": round(float(np.var(movements)), 4),
            "events":            events[:15]
        }

    # ─────────────────────────────────────────────────────────────
    #  Compile
    # ─────────────────────────────────────────────────────────────
    def _compile(self, gaze_log, pose_log, hand_log, blink_log, duration):
        # ── Gaze stats ──────────────────────────────────────────
        if gaze_log:
            n     = len(gaze_log)
            down  = sum(1 for g in gaze_log if g["gaze"] == "down") / n
            cam   = sum(1 for g in gaze_log if g["gaze"] == "camera") / n
            away  = 1 - down - cam
            pitches = [g["head_pitch"] for g in gaze_log]
            gaze_r = {
                "down_ratio":    round(down, 2),
                "camera_ratio":  round(cam, 2),
                "away_ratio":    round(max(0, away), 2),
                "avg_head_pitch":round(float(np.mean(pitches)), 1),
                "head_pitch_std":round(float(np.std(pitches)), 1),
                "events": [g for g in gaze_log if g["gaze"] != "camera"][:20]
            }
        else:
            gaze_r = {"down_ratio": 0, "camera_ratio": 0, "away_ratio": 0,
                      "avg_head_pitch": 0, "head_pitch_std": 0, "events": []}

        posture_r = self._analyze_posture(pose_log)
        hand_r    = self._analyze_hands(hand_log)

        # 종합 비언어 점수
        cam_score  = gaze_r["camera_ratio"] * 10
        post_score = posture_r["posture_score"]
        hand_ok    = max(0, 10 - hand_r["fidget_score"])
        overall    = cam_score * 0.4 + post_score * 0.35 + hand_ok * 0.25

        return {
            "gaze":                   gaze_r,
            "posture":                posture_r,
            "hands":                  hand_r,
            "blink_count":            len(blink_log),
            "overall_nonverbal_score":round(overall, 1),
            "duration":               round(duration, 1),
            "gaze_timeline":          gaze_log
        }
