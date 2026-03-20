"""
통합 안전 감지 시스템
====================
1단계  안전벨트 감지
  - no-seatbelt 감지 시 → 졸음감지 비활성 (대기)
  - seatbelt 가 SEATBELT_THRESHOLD 초 이상 연속 감지 → 졸음감지 활성화

2단계  졸음운전 감지 (안전벨트 조건 충족 후에만 동작)
  - YOLOv8-face 로 얼굴 감지 → crop
  - 눈 모델로 open/closed 판별
  - closed_eye 가 ALARM_SECS 초 이상 지속 → 경보

[조작]
  q / ESC : 종료
  Space   : 일시정지 / 재개
"""



import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import numpy as np
from ultralytics import YOLO


# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR         = r"C:\PROJECT6"

SEATBELT_WEIGHTS = rf"{BASE_DIR}\models\seatbelt.pt"
FACE_WEIGHTS     = rf"{BASE_DIR}\models\yolov8n-face.pt"
EYE_WEIGHTS      = rf"{BASE_DIR}\models\sleep_detect.pt"

VIDEO_SOURCE     = rf"{BASE_DIR}\test_video\sleep_test3.mp4"
#VIDEO_SOURCE     = 0                                         # 0 이면 웹캠

OUTPUT_DIR       = rf"{BASE_DIR}\result_video"               # 결과 영상 저장 폴더
# ──────────────────────────────────────────────────────────

# ── 감지 설정 ──────────────────────────────────────────────
SEATBELT_CONF      = 0.4
SEATBELT_THRESHOLD = 5.0      # 안전벨트 연속 감지 필요 시간(초) → 졸음감지 ON

FACE_CONF   = 0.4
EYE_CONF    = 0.35
FACE_PAD    = 60              # 얼굴 crop 여유 픽셀
ALARM_SECS  = 2               # closed_eye 연속 감지 → 경보(초)

DISPLAY_SIZE   = (450, 800)   # (max_width, max_height)
BLINK_INTERVAL = 0.35         # 경고 깜빡임 속도 
# ──────────────────────────────────────────────────────────

# ── 색상 (BGR) ─────────────────────────────────────────────
CLR_GREEN  = (50, 220, 50)
CLR_RED    = (30,  30, 255)
CLR_ORANGE = (0,  165, 255)
CLR_YELLOW = (0,  230, 230)
CLR_GRAY   = (160, 160, 160)
CLR_FACE   = (255, 200,   0)
CLOSED_ID  = 0
CLS_NAME   = {0: "closed", 1: "open"}
# ──────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════
# 1.  안전벨트 감지
# ════════════════════════════════════════════════════════════

def detect_seatbelt(frame, sb_model):
    """프레임에서 안전벨트 감지 후 박스 그리기"""
    
    results = sb_model.predict(frame, conf=SEATBELT_CONF, verbose=False)
    result  = results[0]
    
    counts  = {"seatbelt": 0, "no-seatbelt": 0}

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            label  = sb_model.names[cls_id]
            counts[label] = counts.get(label, 0) + 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 박스 + 텍스트 그리기
            color   = CLR_GREEN if label == "seatbelt" else CLR_RED
            display = "Seatbelt" if label == "seatbelt" else "NoSeatbelt"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{display} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return counts


# ════════════════════════════════════════════════════════════
# 2.  졸음 감지 (2단계: 얼굴 → 눈)
# ════════════════════════════════════════════════════════════

# 현재는 운전자 1명만 감지 하도록 구현 했음
# 다인용으로 구현 하려면
# 안전벨트가 누구의 것인지 mapping이 필요로 하고, 현재 안전벨트 착용 시간을 기준으로 eye_detect으로 넘어가는데 이 기준을 잡기 힘듦
# 추후에 App을 만들 땐 다인으로 가능

def get_best_face(face_results, img_w, img_h):
    """신뢰도 가장 높은 얼굴 박스 반환 (패딩 포함). 없으면 None."""
    
    if face_results.boxes is None or len(face_results.boxes) == 0:
        return None
    best = max(face_results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    x1 = max(0,     x1 - FACE_PAD)
    y1 = max(0,     y1 - FACE_PAD)
    x2 = min(img_w, x2 + FACE_PAD)
    y2 = min(img_h, y2 + FACE_PAD)
    return x1, y1, x2, y2


def draw_eye_boxes(frame, boxes, offset_x, offset_y):
    """눈 박스를 원본 좌표로 변환하여 그리기."""
    has_closed = False
    n_open = n_closed = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 += offset_x;  x2 += offset_x
        y1 += offset_y;  y2 += offset_y
        color = CLR_RED if cls_id == CLOSED_ID else CLR_GREEN
        label = f"{CLS_NAME.get(cls_id, cls_id)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        if cls_id == CLOSED_ID:
            has_closed = True
            n_closed  += 1
        else:
            n_open += 1
    return has_closed, n_open, n_closed


def detect_eyes(frame, face_model, eye_model):
    """
    1) 얼굴 감지 → crop
    2) crop 영역에서 눈 상태 감지
    반환: (has_closed, n_open, n_closed)
    """
    h, w = frame.shape[:2]              # (H, W, C)
    face_res = face_model(frame, conf=FACE_CONF, verbose=False)[0]      # 전체 프레임에서 얼굴 감지
    face_box = get_best_face(face_res, w, h)                            # get_best_face()로 신뢰도 1위 얼굴 좌표 변환

    if face_box is None:                                     
        # 얼굴 미감지 → 전체 프레임 fallback
        eye_res = eye_model(frame, conf=EYE_CONF, verbose=False)[0]
        return draw_eye_boxes(frame, eye_res.boxes, 0, 0)

    fx1, fy1, fx2, fy2 = face_box                                       # 얼굴 박스 그리기
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (90, 90, 90), 1)


    ## 중요 #########################################################
    face_crop = frame[fy1:fy2, fx1:fx2]                                 # numpy 슬라이싱으로 얼굴 영역만 잘라냄
    eye_res   = eye_model(face_crop, conf=EYE_CONF, verbose=False)[0]
    return draw_eye_boxes(frame, eye_res.boxes, fx1, fy1)

"""
얼굴 감지
    │
    ├─ 얼굴 없음 → 전체 프레임으로 눈 감지 (fallback)
    │
    └─ 얼굴 있음
          ↓
       frame[fy1:fy2, fx1:fx2]  ← 얼굴 영역만 crop
          ↓
       eye_model(face_crop)     ← crop된 이미지에서 눈 감지
          ↓
       draw_eye_boxes()         ← 원본 좌표로 변환해서 박스 그리기
       
    crop - 전체 프레임에서 눈을 찾으면 배경 물체를 눈으로 오탐할 수 있어서, 얼굴 영역만 잘라서 정확도를 높임
"""


# ════════════════════════════════════════════════════════════
# 3.  HUD 표시
# ════════════════════════════════════════════════════════════

def draw_hud(frame, sb_counts, sb_elapsed, sleep_active,
             elapsed_closed, is_alert, blink):
    h, w = frame.shape[:2]

    # 하단 반투명 바
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    if not sleep_active:
        # ── Phase 1: 안전벨트 상태 ──────────────────────────
        sb   = sb_counts.get("seatbelt", 0)
        nosb = sb_counts.get("no-seatbelt", 0)

        if nosb > 0:
            sb_color = CLR_RED
            sb_text  = "NO SEATBELT  (WARNING)"
        elif sb > 0:
            sb_color = CLR_GREEN
            sb_text  = f"SEATBELT OK  [{sb_elapsed:.1f}s / {SEATBELT_THRESHOLD:.0f}s]"
        else:
            sb_color = CLR_GRAY
            sb_text  = "SEATBELT: DETECTING..."

        cv2.putText(frame, sb_text,
                    (12, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, sb_color, 2, cv2.LINE_AA)

        # 프로그레스 바
        if sb > 0 and nosb == 0:
            ratio = min(sb_elapsed / SEATBELT_THRESHOLD, 1.0)
            bar_w = int(w * ratio)
            cv2.rectangle(frame, (0, h - 73), (w,     h - 70), (50, 50, 50), -1)
            cv2.rectangle(frame, (0, h - 73), (bar_w, h - 70), CLR_YELLOW,   -1)

    else:
        # ── Phase 2: 졸음 감지 상태 ─────────────────────────
        if is_alert:
            sleep_color = CLR_RED
            sleep_text  = "!! WARNING !!"
        elif elapsed_closed > 0:
            sleep_color = CLR_ORANGE
            sleep_text  = f"EYE DETECT: CAUTION  [{elapsed_closed:.1f}s]"
        else:
            sleep_color = CLR_GREEN
            sleep_text  = "EYE DETECT: AWAKE"

        cv2.putText(frame, sleep_text, 
                    (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sleep_color, 2, cv2.LINE_AA)

        # 졸음 경보 테두리 + 깜빡이는 텍스트
        if is_alert and blink:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), CLR_RED, 6)
            txt   = "!! WARNING !!"
            scale, thick = 1.5, 4
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
            x, y = (w - tw) // 2, th + 20
            cv2.putText(frame, txt, (x + 2, y + 2),
                        cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, scale, CLR_RED, thick, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════
# 4.  메인 루프
# ════════════════════════════════════════════════════════════

def run():
    print("=" * 58)
    print("  통합 안전 감지 시스템  (안전벨트 → 졸음감지)")
    print("=" * 58)
    print("  모델 로딩 중...")

    sb_model   = YOLO(SEATBELT_WEIGHTS)
    face_model = YOLO(FACE_WEIGHTS)
    eye_model  = YOLO(EYE_WEIGHTS)
    print("  모델 로드 완료\n")

    source = VIDEO_SOURCE
    cap    = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        print(f"  영상을 열 수 없습니다: {source}")
        return

    orig_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = max(1, int(1000 / orig_fps))
    orig_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  소스   : {source}")
    print(f"  임계값 : 안전벨트 {SEATBELT_THRESHOLD}초 → 졸음감지 ON")
    print("  [q / ESC: 종료  |  Space: 일시정지]\n")

    # 결과 영상 저장 설정
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if isinstance(source, str):
        src_name = os.path.splitext(os.path.basename(source))[0]
    else:
        src_name = "webcam"
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    out_path    = os.path.join(OUTPUT_DIR, f"{src_name}_{timestamp}.mp4")
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer  = cv2.VideoWriter(out_path, fourcc, orig_fps, (orig_w, orig_h))
    print(f"  결과 저장 : {out_path}\n")

    # 상태 변수
    sb_start     = None    # 안전벨트 연속 감지 시작 시각
    sb_elapsed   = 0.0
    sleep_active = False

    closed_start = None    # closed_eye 연속 감지 시작 시각
    elapsed_cls  = 0.0

    blink_state = False
    blink_last  = time.time()
    paused      = False
    frame       = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            # ── Phase 1: 안전벨트 감지 ───────────────────────
            sb_counts = {"seatbelt": 0, "no-seatbelt": 0}
            is_alert  = False

            if not sleep_active:
                sb_counts = detect_seatbelt(frame, sb_model)
                sb   = sb_counts.get("seatbelt",    0)
                nosb = sb_counts.get("no-seatbelt", 0)

                if sb > 0 and nosb == 0:
                    if sb_start is None:
                        sb_start = now
                    sb_elapsed = now - sb_start
                    if sb_elapsed >= SEATBELT_THRESHOLD:
                        sleep_active = True
                        print("  안전벨트 확인 완료 → 졸음감지 시작")
                else:
                    sb_start   = None
                    sb_elapsed = 0.0

            # ── Phase 2: 졸음 감지 (안전벨트 감지 비활성) ───
            else:
                has_closed, _, _ = detect_eyes(frame, face_model, eye_model)

                if has_closed:
                    if closed_start is None:
                        closed_start = now
                    elapsed_cls = now - closed_start
                else:
                    closed_start = None
                    elapsed_cls  = 0.0

                is_alert = elapsed_cls >= ALARM_SECS

            # 블링크 타이머
            if now - blink_last >= BLINK_INTERVAL:
                blink_state = not blink_state
                blink_last  = now

            draw_hud(frame, sb_counts, sb_elapsed, sleep_active,
                     elapsed_cls, is_alert, blink_state)
            out_writer.write(frame)

        # 화면 출력 (일시정지 중에도 마지막 프레임 유지)
        if frame is not None:
            if DISPLAY_SIZE:
                fh, fw = frame.shape[:2]
                max_w, max_h = DISPLAY_SIZE
                scale = min(max_w / fw, max_h / fh)
                disp  = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                disp = frame
            cv2.imshow("Safety Detection  [q:종료 | Space:일시정지]", disp)

        key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused

    out_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"  결과 영상 저장 완료: {out_path}")
    print("  종료되었습니다.")


if __name__ == "__main__":
    run()



