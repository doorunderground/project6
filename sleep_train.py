"""
졸음운전 감지 모델 학습 스크립트

[감지 대상]
  open_eye   (class 0) → 눈 뜸  = 정상
  closed_eye (class 1) → 눈 감음 = 졸음 위험

[데이터셋 구조]
  C:/PROJECT6/
    data/
      train/
        images/   ← 학습 이미지
        labels/   ← YOLO 세그멘테이션 형식 라벨 (.txt)
      valid/
        images/   ← 검증 이미지
        labels/   ← YOLO 세그멘테이션 형식 라벨 (.txt)
    data.yaml     ← 클래스 정보 및 경로 설정
"""

# ============================================================
# 0. Import
# ============================================================

import os
import yaml
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# 1. 학습 하이퍼파라미터 설정
# ============================================================

# ── 모델 선택 ────────────────────────────────────────────────
MODEL      = r"C:\sleepdetect\runs\detect\runs\train\sleep_detect8\weights\best.pt"  # yolov8.pt

# ── 데이터셋 설정 파일 ────────────────────────────────────────
DATA_YAML  = "data/data.yaml"

# ── 학습 파라미터 ─────────────────────────────────────────────
EPOCHS     = 200        # 전체 학습 반복 횟수
IMG_SIZE   = 640        # 입력 이미지 크기 (YOLO 기본값 640×640)
BATCH_SIZE = 32         # 한 번에 처리할 이미지 수
# WORKERS  = 8          # 생략해도 YOLO 기본 = 8 (자동 적용)
DEVICE     = 0          # GPU

# ── 학습률 (Learning Rate) ────────────────────────────────────
LR0        = 1e-3       # 학습 초반 속도 (빠르게 대략적 학습)
LRF        = 0.01       # 최종 학습률 = LR0 × LRF = 0.00001로 끝남 (느리게 정밀 마무리)

# ── 조기 종료 ─────────────────────────────────────────────────
PATIENCE   = 50

# ── 결과 저장 경로 ────────────────────────────────────────────
PROJECT_DIR = "runs/train"
RUN_NAME    = "sleep_detect9"





# ── 데이터 증강 (Augmentation) ─────────────────────────────────


AUGMENT_CFG = {
    # H S V / Hue(색조), Saturation(채도), Value(밝기)
    
    "hsv_h"   : 0.015,   # default
    "hsv_s"   : 0.7,     # 
    "hsv_v"   : 0.4,     # 밝기 (낮 / 밤 / 터널 / 흐린 날)

    "degrees" : 5.0,     # 미세 각도 오차 대응
    "fliplr"  : 0.5,     # 좌우 반전 (눈 좌우 대칭)
    "flipud"  : 0.0,     # 상하 반전 없음 (눈이 뒤집어지는 상황은 없음)

    "mosaic"  : 1.0,     # 이미지 4장을 하나로 합쳐서 새 학습 이미지를 생성 (100%)
}

    # translate ?
    # scale     ? 
    # mixup     ?


# ============================================================
# 2. 학습 실행 함수
# ============================================================

'''
        1 epoch 흐름
    1. train 이미지로 학습
    2. on_train_epoch_end 호출  ← train_loss 저장
    3. valid 이미지로 검증
    4. on_epoch_end 호출  ← 출력
'''

_train_loss = 0.0

def on_train_epoch_end(trainer):
    global _train_loss                              # epoch의 배치 평균 손실값(box/class/dfl 합산)
    if trainer.tloss is not None:
        _train_loss = float(trainer.tloss.mean()) if hasattr(trainer.tloss, "mean") else float(trainer.tloss)


def on_epoch_end(trainer):
    global _train_loss
    epoch = trainer.epoch + 1
    total = trainer.epochs
    m     = trainer.metrics

    # 검증 손실 계산
    # YOLO의 손실 구성 : box/class/dfl
    val_loss = (m.get("val/box_loss", 0)           # 바운딩박스 위치가 얼마나 틀렸는지
              + m.get("val/cls_loss", 0)           # 클래스 분류가 얼마나 틀렸는지 (눈 뜸/감음)
              + m.get("val/dfl_loss", 0))          # 박스 경계선 정밀도가 얼마나 틀렸는지
    acc      = m.get("metrics/mAP50(B)", 0)
    # mAP50 - IoU(예측 박스와 정답 박스가 얼마나 겹치는지) 50% 이상 겹치면 → 맞게 탐지했다.

    # Print
    # ex) [████████░░] 80%
    filled = int(20 * epoch / total)
    bar    = "█" * filled + "░" * (20 - filled)

    print(f"\nTrain")
    print(f"┌─ Epoch {epoch}/{total}  [{bar}]  {epoch/total*100:.0f}%")
    print(f"│  Loss : {_train_loss:.4f}")
    print(f"└─ Acc  : {acc:.4f}")
    print(f"\n Valid")
    print(f"   Epoch : {epoch}/{total}          Loss : {val_loss:.4f}          Acc : {acc:.4f}\n")


def print_dataset_info():
    """학습 시작 전 데이터셋 정보 출력"""
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    print("=" * 55)
    print(f"  데이터 경로 : {cfg.get('path', '.')}")
    print(f"  클래스 수   : {cfg['nc']}")
    print(f"  클래스 목록 : {cfg['names']}")
    train_dir = os.path.normpath(os.path.join(cfg.get("path", "."), cfg["train"], "..", "labels"))
    valid_dir = os.path.normpath(os.path.join(cfg.get("path", "."), cfg["val"],   "..", "labels"))
    if os.path.exists(train_dir):
        print(f"  학습 샘플 수: {len(os.listdir(train_dir))}")
    if os.path.exists(valid_dir):
        print(f"  검증 샘플 수: {len(os.listdir(valid_dir))}")
    print("=" * 55)


# 학습
def train():
    print_dataset_info()

    model = YOLO(MODEL)    # 이전 학습 가중치 로드 (이어학습)

    # ── epoch 종료마다 요약 출력 콜백 등록 ─────────────────────
    model.add_callback("on_train_epoch_end", on_train_epoch_end)  # train loss 저장
    model.add_callback("on_fit_epoch_end",   on_epoch_end)        # 요약 출력

    # ── 모델 학습 ─────────────────────────────────────────────
    results = model.train(
        data    = DATA_YAML,        # 데이터셋 설정 파일
        epochs  = EPOCHS,           # 총 학습 epoch
        imgsz   = IMG_SIZE,         # 입력 이미지 크기
        batch   = BATCH_SIZE,       # 배치 크기
    #   workers = WORKERS
        device  = DEVICE,           # 학습 디바이스
        verbose = False,            # YOLO 기본 출력 끔 (콜백으로 대체)


        # 학습률 설정
        optimizer     = "AdamW",   
        lr0           = LR0,
        lrf           = LRF,

        patience      = PATIENCE,   

        weight_decay  = 0.0005,     # 과적합 방지
        warmup_epochs = 3,          # 처음 3 epoch 워밍업

        # 결과 저장 경로
        project    = PROJECT_DIR,
        name       = RUN_NAME,

        # 체크포인트: best.pt 와 last.pt 모두 저장
        save        = True,
        save_period = 10,           # 10 epoch 마다 중간 가중치 저장

        # 학습 모니터링
        plots = True,               # 학습 곡선, confusion matrix 등 시각화

        # 데이터 증강 설정
        **AUGMENT_CFG,
        ## '**' - 딕셔너리를 풀어서 개별 인자로 전달하는 파이썬 문법
    )


    # ── 학습 완료 후 best.pt 경로 출력 ───────────────────────
    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("  학습 완료!")
    print(f"  최고 가중치 저장 위치:\n  {best_weights}")
    print("=" * 55)

    return results


# ============================================================
# 3. valid 함수 (학습 후 성능 확인)  학습 다 끝낸 best.pt로 (한 번만 실행)
# ============================================================

def validate(weights_path: str = None):

    # best.pt
    if weights_path is None:
        weights_path = str(
            Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
        )

    if not os.path.exists(weights_path):
        print(f"[오류] 가중치 파일 없음: {weights_path}")
        return

    print(f"\n검증 시작: {weights_path}")
    model   = YOLO(weights_path)                               # best.pt
    metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE)        # 검증 실행

    p   = metrics.box.mp     # Precision
    r   = metrics.box.mr     # Recall
    m50 = metrics.box.map50  # mAP50
    m95 = metrics.box.map    # mAP50-95

    print("\n" + "=" * 55)
    print("  최종 검증 결과 (best.pt 기준)")
    print("=" * 55)
    print(f"  Acc       : {m50:.4f}")
    print(f"  mAP50-95  : {m95:.4f}  (IoU 50~95% 구간 평균)")
    print(f"  Precision : {p:.4f}  탐지한 것 중 {p*100:.1f}% 가 맞음")
    print(f"  Recall    : {r:.4f}  실제 객체의 {r*100:.1f}% 를 찾아냄")
    print("=" * 55)

"""
    학습 중 (매 epoch)
    - on_epoch_end 콜백 -> valid 데이터로 실시간 모니터링용 검증
    
    학습 완료 후(한 번만)
    - validate() -> best.pt(학습 중 가장 성능 좋았던 가중치)를 불러와서 최종 검증
"""



# ============================================================
# 4. 진입점
# ============================================================

if __name__ == "__main__":
    # Step 1: 학습 실행
    results = train()

    # Step 2: 학습 완료 후 자동 검증 (belt_train 과 달리 자동 실행)
    validate(weights_path=str(Path(results.save_dir) / "weights" / "best.pt"))
    
    