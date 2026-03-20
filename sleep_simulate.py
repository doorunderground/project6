"""
졸음운전 감지 시뮬레이션 테스트
- 1단계: YOLOv8-face 로 얼굴 감지 → crop
- 2단계: crop된 얼굴에서 눈 감지
- 영상 비율 무관하게 일정한 감지 성능

조작:
  Q          : 종료
  Space      : 일시정지 / 재개
  ← →        : 이미지 시뮬레이션에서 이전/다음 프레임 (방향키)
"""

import cv2
import os
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────
FACE_WEIGHTS  = r"C:\sleepdetect\yolov8n-face.pt"
WEIGHTS       = r"C:\PROJECT6\sleep_best.pt"
VALID_IMG_DIR = r"data/valid/images"

VIDEO_SOURCE  = r'C:\PROJECT6\test_video\sleep_test1.mp4'

CONF          = 0.35    # 눈 감지 신뢰도 임계값
FACE_CONF     = 0.4    # 얼굴 감지 신뢰도 임계값
FACE_PAD      = 60     # 얼굴 crop 여유 픽셀
ALARM_SECS    = 1.5    # closed_eye 연속 감지 → 경보 발생(초)
SIM_FPS       = 4
SAVE_VIDEO    = True
OUTPUT_PATH   = r"C:\PROJECT6\result_video"
DISPLAY_SIZE  = (800, 800)
# ────────────────────────────────────────────────────

CLOSED_ID  = 0
CLS_NAME   = {0: "closed_eye", 1: "open_eye"}
CLR_OPEN   = (50, 220, 50)
CLR_CLOSED = (30, 30, 255)
CLR_FACE   = (255, 200, 0)    # 얼굴 박스 색 (하늘색)
CLR_OK     = (50, 220, 50)
CLR_WARN   = (0, 165, 255)
CLR_ALERT  = (30, 30, 255)


# ── 감지 헬퍼 ─────────────────────────────────────────

def get_best_face(face_results, img_w, img_h):
    """가장 신뢰도 높은 얼굴 박스 반환. 없으면 None."""
    if face_results.boxes is None or len(face_results.boxes) == 0:
        return None
    best = max(face_results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    # 패딩 적용 + 클램프
    x1 = max(0,     x1 - FACE_PAD)
    y1 = max(0,     y1 - FACE_PAD)
    x2 = min(img_w, x2 + FACE_PAD)
    y2 = min(img_h, y2 + FACE_PAD)
    return x1, y1, x2, y2


def detect_eyes(frame, face_model, eye_model):
    """
    2단계 감지:
      1) face_model 로 얼굴 crop
      2) eye_model 로 눈 감지
    반환: (has_closed, n_open, n_closed, face_box or None)
    """
    h, w = frame.shape[:2]

    # 1단계 - 얼굴 감지
    face_res  = face_model(frame, conf=FACE_CONF, verbose=False)[0]
    face_box  = get_best_face(face_res, w, h)

    if face_box is None:
        # 얼굴 미감지 → 전체 프레임으로 눈 감지 (fallback)
        eye_res = eye_model(frame, conf=CONF, verbose=False)[0]
        has_closed, n_open, n_closed = draw_eye_boxes(frame, eye_res.boxes, 0, 0)
        return has_closed, n_open, n_closed, None

    fx1, fy1, fx2, fy2 = face_box

    # 2단계 - 얼굴 crop 후 눈 감지
    face_crop = frame[fy1:fy2, fx1:fx2]
    eye_res   = eye_model(face_crop, conf=CONF, verbose=False)[0]

    # 원본 프레임 좌표로 변환하여 그리기
    has_closed, n_open, n_closed = draw_eye_boxes(frame, eye_res.boxes, fx1, fy1)

    return has_closed, n_open, n_closed, face_box


def draw_eye_boxes(frame, boxes, offset_x, offset_y):
    """눈 박스를 offset 적용하여 원본 프레임에 그리기."""
    has_closed = False
    n_open = n_closed = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # offset 적용 (crop 좌표 → 원본 좌표)
        x1 += offset_x; x2 += offset_x
        y1 += offset_y; y2 += offset_y

        color = CLR_CLOSED if cls_id == CLOSED_ID else CLR_OPEN
        label = f"{CLS_NAME.get(cls_id, cls_id)}  {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        if cls_id == CLOSED_ID:
            has_closed = True
            n_closed  += 1
        else:
            n_open += 1

    return has_closed, n_open, n_closed


def draw_face_box(frame, face_box):
    """얼굴 박스 그리기."""
    if face_box is None:
        return
    x1, y1, x2, y2 = face_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_FACE, 2)
    cv2.putText(frame, "face", (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_FACE, 1, cv2.LINE_AA)


# ── HUD / 경보 ────────────────────────────────────────

def draw_hud(frame, frame_idx, total, fps, elapsed_closed, is_alert, n_open, n_closed, blink):
    h, w = frame.shape[:2]
    bar_h  = 72
    prog_h = 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    if is_alert:
        s_clr, s_txt = CLR_ALERT, "DROWSY"
    elif elapsed_closed > 0:
        s_clr, s_txt = CLR_WARN, "CAUTION"
    else:
        s_clr, s_txt = CLR_OK, "AWAKE"

    cv2.putText(frame, f"STATUS : {s_txt}",
                (14, h - bar_h + 26), cv2.FONT_HERSHEY_DUPLEX, 0.72, s_clr, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Closed Eye : {elapsed_closed:.1f} s",
                (14, h - bar_h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)

    mid = w // 2
    cv2.putText(frame, f"open : {n_open}   closed : {n_closed}",
                (mid - 100, h - bar_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS : {fps:.1f}",
                (mid - 50, h - bar_h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 140, 140), 1, cv2.LINE_AA)

    f_txt = f"{frame_idx + 1} / {total}"
    (fw, _), _ = cv2.getTextSize(f_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, f_txt,
                (w - fw - 14, h - bar_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1, cv2.LINE_AA)

    prog_y = h - bar_h - prog_h
    cv2.rectangle(frame, (0, prog_y), (w, prog_y + prog_h), (50, 50, 50), -1)
    fill_w = int(w * (frame_idx + 1) / max(total, 1))
    cv2.rectangle(frame, (0, prog_y), (fill_w, prog_y + prog_h), s_clr, -1)

    if is_alert and blink:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), CLR_ALERT, 6)


def draw_alert_text(frame, blink):
    if not blink:
        return
    _, w = frame.shape[:2]
    txt = "!! DROWSINESS ALERT !!"
    scale, thick = 1.1, 3
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    x, y = (w - tw) // 2, th + 18
    cv2.putText(frame, txt, (x + 2, y + 2),
                cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, scale, CLR_ALERT, thick, cv2.LINE_AA)


# ── 메인 ─────────────────────────────────────────────

def _find_video_in_dir(folder: str):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None
    for f in sorted(folder_path.iterdir()):
        if f.suffix.lower() in exts:
            return str(f)
    return None


def run(source=None):
    if source and os.path.isdir(source):
        found = _find_video_in_dir(source)
        source = found if found else None

    is_video = source and os.path.isfile(source)

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"파일을 열 수 없습니다: {source}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or SIM_FPS
        print(f"[동영상] {source}  |  프레임: {total_frames}  FPS: {src_fps:.1f}")
    else:
        exts      = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = sorted(p for p in Path(VALID_IMG_DIR).iterdir()
                           if p.suffix.lower() in exts)
        if not img_paths:
            print(f"이미지가 없습니다: {VALID_IMG_DIR}")
            return
        total_frames = len(img_paths)
        src_fps      = SIM_FPS
        print(f"[이미지 시뮬레이션] {VALID_IMG_DIR}  |  {total_frames}장")

    # ── 모델 로드 ─────────────────────────────────────
    if not os.path.exists(WEIGHTS):
        print(f"가중치 파일 없음: {WEIGHTS}")
        return

    face_model = YOLO(FACE_WEIGHTS)   # 자동 다운로드 or 로컬
    eye_model  = YOLO(WEIGHTS)
    print(f"  얼굴 모델 : {FACE_WEIGHTS}")
    print(f"  눈   모델 : {WEIGHTS}")
    print("\n[조작] Q=종료  Space=일시정지  ←→=프레임이동(이미지모드)\n")

    # ── 상태 변수 ─────────────────────────────────────
    writer        = None
    frame_delay   = max(1, int(1000 / src_fps))
    closed_start  = None
    elapsed_cls   = 0.0
    total_open    = total_closed = alert_events = 0
    fps_val       = 0.0
    fps_frame_cnt = 0
    fps_t         = time.time()
    blink_state   = False
    blink_last    = time.time()
    BLINK_INTERVAL = 0.35
    paused        = False
    frame_idx     = 0

    # ── 루프 ─────────────────────────────────────────
    while True:

        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            frame = cv2.imread(str(img_paths[frame_idx]))
            if frame is None:
                frame_idx += 1
                continue

        if SAVE_VIDEO and writer is None:
            h_f, w_f = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, max(src_fps, 5), (w_f, h_f))

        # ── 2단계 감지 ────────────────────────────────
        has_closed, n_open, n_closed, face_box = detect_eyes(frame, face_model, eye_model)

        draw_face_box(frame, face_box)
        total_open   += n_open
        total_closed += n_closed

        # ── 졸음 타이머 ───────────────────────────────
        now = time.time()
        if has_closed:
            if closed_start is None:
                closed_start = now
            elapsed_cls = now - closed_start
        else:
            closed_start = None
            elapsed_cls  = 0.0

        is_alert = elapsed_cls >= ALARM_SECS

        if now - blink_last >= BLINK_INTERVAL:
            blink_state = not blink_state
            blink_last  = now
            if is_alert and blink_state:
                alert_events += 1

        fps_frame_cnt += 1
        elapsed_fps = now - fps_t
        if elapsed_fps >= 1.0:
            fps_val       = fps_frame_cnt / elapsed_fps
            fps_t         = now
            fps_frame_cnt = 0

        if is_alert:
            draw_alert_text(frame, blink_state)

        draw_hud(frame, frame_idx, total_frames, fps_val,
                 elapsed_cls, is_alert, n_open, n_closed, blink_state)

        if writer:
            writer.write(frame)

        if DISPLAY_SIZE:
            fh, fw = frame.shape[:2]
            max_w, max_h = DISPLAY_SIZE
            scale = min(max_w / fw, max_h / fh)
            disp  = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        else:
            disp = frame
        cv2.imshow("Drowsiness Simulation  [Q: 종료 | Space: 일시정지]", disp)

        key = cv2.waitKey(1 if paused else frame_delay) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 or key == 2424832:
            if not is_video:
                frame_idx = max(0, frame_idx - 1)
                closed_start = None
                elapsed_cls  = 0.0
            continue
        elif key == 83 or key == 2555904:
            if not is_video and paused:
                frame_idx += 1
        elif key == 255 and paused:
            continue

        if not paused:
            frame_idx += 1

        if not is_video and frame_idx >= total_frames:
            break

    if is_video:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("  [시뮬레이션 결과 요약]")
    print("=" * 50)
    print(f"  처리 프레임        : {frame_idx}")
    print(f"  open_eye 탐지 수   : {total_open}")
    print(f"  closed_eye 탐지 수 : {total_closed}")
    print(f"  경보 발생 횟수     : {alert_events}")
    if writer:
        print(f"  결과 영상          : {OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="졸음운전 감지 시뮬레이션 테스트")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--fps",    type=int,   default=SIM_FPS)
    parser.add_argument("--conf",   type=float, default=CONF)
    args = parser.parse_args()
    SIM_FPS = args.fps
    CONF    = args.conf
    source  = args.source or VIDEO_SOURCE
    run(source=source)
