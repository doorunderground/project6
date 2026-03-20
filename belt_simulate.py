"""
detect_video.py
===============
학습된 YOLOv8/YOLO11 가중치로 시뮬레이션 영상에서
안전벨트 착용 여부를 실시간 감지하는 스크립트

[사용 방법]
  1. 기본 실행 (videos/test_video.mp4 실시간 재생)
       python detect_video.py

  2. 영상 직접 지정
       python detect_video.py --video 영상.mp4

  3. 신뢰도 임계값 조정 (기본 0.4)
       python detect_video.py --conf 0.5

  4. 재생 중 단축키
       q : 종료
       Space : 일시정지 / 재개

[출력]
  - OpenCV 창에서 실시간 감지 결과 표시
  - 콘솔: 종료 후 전체 통계 출력
"""

# ============================================================
# 0. 패키지 임포트
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. 설정 상수
# ============================================================

# 가중치 경로 
DEFAULT_WEIGHTS = r"C:\PROJECT6\belt_simulate.py"

# 테스트 영상 
DEFAULT_VIDEO = r"C:\PROJECT6\test_video\belt_test1.mp4"

# 결과 저장 폴더
OUTPUT_DIR = r"C:\PROJECT6\result_video"

# 감지 설정
DEFAULT_CONF   = 0.4    # 신뢰도 임계값 (0~1, 낮을수록 더 많이 감지)
DEFAULT_IOU    = 0.45   # NMS IoU 임계값
IMG_SIZE       = 640    # 추론 이미지 크기 (학습 시와 동일하게)

# 화면 표시 크기 (None이면 비율 자동 계산 또는 원본 유지)
DISPLAY_WIDTH  = 800    # 표시 창 너비 (픽셀)
DISPLAY_HEIGHT = 800   # 표시 창 높이 (픽셀), None이면 너비 기준 비율 자동 계산

# 클래스별 시각화 색상 (BGR)
CLASS_COLORS = {
    "seatbelt"    : (50, 205, 50),    # 초록색 → 안전벨트 착용
    "no-seatbelt" : (0, 60, 255),     # 빨간색 → 미착용 (경고)
}

# 상태 패널 배경색
PANEL_COLOR_OK   = (20, 120, 20)    # 착용: 어두운 초록
PANEL_COLOR_WARN = (20, 20, 160)    # 미착용: 어두운 빨강
PANEL_COLOR_NONE = (50, 50, 50)     # 미감지: 회색


# ============================================================
# 2. 헬퍼 함수
# ============================================================

DISPLAY_NAMES = {
    "seatbelt"   : "Seatbelt",
    "no-seatbelt": "NoSeatbelt",
}

def draw_detections(frame: np.ndarray, result, class_names: list) -> tuple[np.ndarray, dict]:
    """
    프레임에 바운딩 박스와 레이블을 그린다.

    Returns
    -------
    annotated_frame : 시각화가 완료된 프레임
    counts          : {'seatbelt': int, 'no-seatbelt': int} 감지 수
    """
    counts = {"seatbelt": 0, "no-seatbelt": 0}

    if result.boxes is None or len(result.boxes) == 0:
        return frame, counts

    for box in result.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        label  = class_names[cls_id]
        counts[label] = counts.get(label, 0) + 1

        # 바운딩 박스 좌표 (픽셀 단위)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(label, (200, 200, 200))

        # 박스 그리기 (두께 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # 레이블 텍스트 (배경 없이 컬러 텍스트)
        display_name = DISPLAY_NAMES.get(label, label)
        text = f"{display_name} {conf:.2f}"
        baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[1]
        cv2.putText(frame, text, (x1, y1 - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

    return frame, counts


def draw_status_panel(frame: np.ndarray, counts: dict, frame_idx: int, fps: float) -> np.ndarray:
    """
    좌측 상단에 현재 감지 상태 요약 패널을 오버레이한다.
    """
    h, w = frame.shape[:2]
    sb    = counts.get("seatbelt", 0)
    no_sb = counts.get("no-seatbelt", 0)

    # 패널 배경
    if no_sb > 0:
        bg_color = PANEL_COLOR_WARN
        status_text = "WARNING: NO SEATBELT"
    elif sb > 0:
        bg_color = PANEL_COLOR_OK
        status_text = "SEATBELT OK"
    else:
        bg_color = PANEL_COLOR_NONE
        status_text = "NO PERSON DETECTED"

    # 반투명 오버레이
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (280, 105), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 상태 텍스트
    cv2.putText(frame, status_text, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # 감지 수
    cv2.putText(frame, f"Seatbelt    : {sb}", (14, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1)
    cv2.putText(frame, f"No-Seatbelt : {no_sb}", (14, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)

    # 프레임 / FPS
    cv2.putText(frame, f"Frame {frame_idx:>5d}  |  {fps:.1f} FPS", (14, 99),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return frame


# ============================================================
# 3. 메인 감지 함수
# ============================================================

def detect_video(
    video_path : str,
    weights    : str   = DEFAULT_WEIGHTS,
    conf       : float = DEFAULT_CONF,
    iou        : float = DEFAULT_IOU,
):
    """
    영상 파일에서 안전벨트 착용 여부를 실시간으로 감지해 화면에 표시한다.

    Parameters
    ----------
    video_path : 입력 영상 경로 (.mp4, .avi, .mov 등)
    weights    : 학습된 YOLO 가중치 파일 경로 (.pt)
    conf       : 감지 신뢰도 임계값 (기본 0.4)
    iou        : NMS IoU 임계값 (기본 0.45)
    """

    # ── 입력 파일 확인 ────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {weights}\n"
            "train_yolo.py 로 먼저 학습을 실행하세요."
        )

    # ── 모델 로드 ─────────────────────────────────────────────
    device = "0" if torch.cuda.is_available() else "cpu"
    print("=" * 55)
    print("  안전벨트 감지 - 영상 추론 시작")
    print("=" * 55)
    print(f"  영상     : {video_path}")
    print(f"  가중치   : {weights}")
    print(f"  신뢰도   : {conf}")
    print(f"  디바이스 : {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")
    print("=" * 55)

    model = YOLO(weights)
    class_names = model.names   # {0: 'no-seatbelt', 1: 'seatbelt'}

    # ── 영상 열기 ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  영상 정보: {vid_w}×{vid_h}, {orig_fps:.1f} FPS, {total_frames} 프레임")
    print("  [재생 중... 'q' 종료  |  Space 일시정지]")

    # 프레임 간 대기 시간 (원본 FPS 속도로 재생)
    frame_delay = max(1, int(1000 / orig_fps))

    # ── 통계 누산기 ───────────────────────────────────────────
    total_counts = {"seatbelt": 0, "no-seatbelt": 0}
    warn_frames  = 0    # no-seatbelt 감지된 프레임 수
    ok_frames    = 0    # seatbelt만 감지된 프레임 수

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # ── YOLO 추론 ──────────────────────────────────────────
            results = model.predict(
                source  = frame,
                conf    = conf,
                iou     = iou,
                imgsz   = IMG_SIZE,
                device  = device,
                verbose = False,
            )

            # ── 시각화 ────────────────────────────────────────────
            result = results[0]
            frame, counts = draw_detections(frame, result, class_names)

            # ── 통계 누산 ─────────────────────────────────────────
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

            if counts.get("no-seatbelt", 0) > 0:
                warn_frames += 1
            elif counts.get("seatbelt", 0) > 0:
                ok_frames += 1

        # ── 화면 표시 (항상 실시간 출력) ─────────────────────
        if DISPLAY_WIDTH or DISPLAY_HEIGHT:
            w = DISPLAY_WIDTH  or frame.shape[1]
            h = DISPLAY_HEIGHT or int(frame.shape[0] * (w / frame.shape[1]))
            disp = cv2.resize(frame, (w, h))
        else:
            disp = frame
        cv2.imshow("Seatbelt Detection  |  q: 종료  Space: 일시정지", disp)
        key = cv2.waitKey(frame_delay) & 0xFF

        if key == ord("q"):
            print("\n  [사용자가 중단했습니다]")
            break
        elif key == ord(" "):
            paused = not paused

    # ── 정리 ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── 최종 통계 출력 ────────────────────────────────────────
    print("\n")
    print("=" * 55)
    print("  감지 완료 - 결과 요약")
    print("=" * 55)
    print(f"  총 seatbelt    감지 수 : {total_counts.get('seatbelt', 0)}")
    print(f"  총 no-seatbelt 감지 수 : {total_counts.get('no-seatbelt', 0)}")
    print(f"  안전벨트 착용 프레임   : {ok_frames}  ({ok_frames/max(ok_frames+warn_frames,1)*100:.1f}%)")
    print(f"  미착용 경고 프레임     : {warn_frames}  ({warn_frames/max(ok_frames+warn_frames,1)*100:.1f}%)")
    print("=" * 55)


# ============================================================
# 4. CLI 진입점
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 안전벨트 감지 - 영상 추론"
    )
    parser.add_argument(
        "--video", "-v", default=DEFAULT_VIDEO,
        help=f"입력 영상 파일 경로 (기본: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--weights", "-w", default=DEFAULT_WEIGHTS,
        help=f"YOLO 가중치 파일 경로 (기본: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=DEFAULT_CONF,
        help=f"감지 신뢰도 임계값 0~1 (기본: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"NMS IoU 임계값 0~1 (기본: {DEFAULT_IOU})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_video(
        video_path = args.video,
        weights    = args.weights,
        conf       = args.conf,
        iou        = args.iou,
    )
