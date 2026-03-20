"""
YOLO 를 사용하여 차량 내부(In-Cabin) 안전벨트 착용 여부를 감지

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Windows 환경에서 PyTorch + numpy 가 OpenMP DLL 을 중복 로드하는 문제 방지

import os
import torch
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# 1. 학습 하이퍼파라미터 설정
# ============================================================

# ── 모델 선택 ────────────────────────────────────────────────
# 차량 내부 고정 카메라 -> 실시간 처리 필요 -> 가벼운 모델 유리

MODEL_NAME   = "yolo11s.pt"

# ── 데이터셋 설정 파일 ────────────────────────────────────────
DATA_YAML    = r"C:\PROJECT6_test\data.yaml"

# ── 학습 파라미터 ─────────────────────────────────────────────
EPOCHS       = 200         # 전체 학습 반복 횟수
IMG_SIZE     = 640         # 입력 이미지 크기 (YOLO 기본값 640×640)
BATCH_SIZE   = 32          # 한 번에 처리할 이미지 수
WORKERS      = 4           # 데이터 로딩 병렬 프로세스 수

# ── 학습률 (Learning Rate) ────────────────────────────────────
LR0          = 0.01        # 학습 초반 속도 (빠르게 대략적 학습)
LRF          = 0.01        # 최종 학습률 = LR0 × LRF = 0.0001로 끝남 (느리게 정밀 마무리)

# ── 조기 종료 ─────────────────────────────────────────────────
PATIENCE     = 50

# ── 결과 저장 경로 ────────────────────────────────────────────
PROJECT_DIR  = r"C:\PROJECT6\runs\detect"
RUN_NAME     = "seatbelt_yolov8"





# ── 데이터 증강 (Augmentation) ─────────────────────────────────

AUGMENT_CFG = {
    # H S V / Hue(색조), Saturation(채도), Value(밝기)
    "hsv_h"       : 1.0,     # 색조 (기본: 0.015)
    "hsv_s"       : 1.0,     # 채도 (기본: 0.7 )  0.0 - 흑백 / 1.0 - 채도 0% ~ 200% 랜덤
    "hsv_v"       : 0.8,     # 밝기 (낮 / 밤 / 터널 /흐린 날) - ±80% 랜덤 변환

    "degrees"     : 5.0,     # 미세 각도 오차 대응
    "translate"   : 0.1,     # 10% 범위에서 상하좌우 이동
    "scale"       : 0.5,     # 원본 크기의 50~150% 범위에서 랜덤 확대/축소

    "fliplr"      : 0.5,     # 좌우 반전 (안전벨트는 좌/우 대칭적으로 착용 가능)

    "mosaic"      : 1.0,     # 이미지 4장을 하나로 합쳐서 새 학습 이미지를 생성 (100%)
    "mixup"       : 0.1,     # 이미지A (안전벨트 착용) + 이미지B (미착용) = 새로운 학습 이미지 (10%) / 경계 상황에서의 훈련
}





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
    global _train_loss                             # epoch의 배치 평균 손실값(box/class/dfl 합산)
    if trainer.tloss is not None:               
        _train_loss = float(trainer.tloss.sum())


def on_epoch_end(trainer): 
    global _train_loss
    epoch = trainer.epoch + 1
    total = trainer.epochs
    m     = trainer.metrics

    # 검증 손실 계산
    # YOLO의 손실 구성 : bos/class/dfl
    val_loss = (m.get("val/box_loss", 0)          # 바운딩박스 위치가 얼마나 틀렸는지
              + m.get("val/cls_loss", 0)          # 클래스 분류가 얼마나 틀렸는지 (착용/미착용)
              + m.get("val/dfl_loss", 0))         # 박스 경셰선 정밀도가 얼마나 틀렸는지
    acc      = m.get("metrics/mAP50(B)", 0)       
    # mAP50 - lou(예측 박스와 정답 박스가 얼마나 겹치는지) 50% 이상 겹치면 -> 맞게 탐지했다. 


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





# 학습
def train():
    
    device = "0" if torch.cuda.is_available() else "cpu"  # GPU check
    print("=" * 55)
    print("  In-Cabin 안전벨트 감지 모델 학습")
    print("=" * 55)
    print(f"  디바이스  : {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == '0' else 'CPU'}")
    print(f"  모델      : {MODEL_NAME}")
    print(f"  데이터    : {DATA_YAML}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Image     : {IMG_SIZE}×{IMG_SIZE}")
    print("=" * 55)

    # ── 데이터셋 파일 존재 확인 ───────────────────────────────
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"data.yaml 을 찾을 수 없습니다: {DATA_YAML}\n"
            "data/ 폴더에 이미지·라벨이 올바르게 배치되어 있는지 확인하세요."
        )
        
    model = YOLO(MODEL_NAME)    # yolo11s.pt 로드

    # ── epoch 종료마다 약 출력 콜백 등록 ─────────────
    model.add_callback("on_train_epoch_end", on_train_epoch_end)  # train loss 저장
    model.add_callback("on_fit_epoch_end",   on_epoch_end)        # 요약 출력

    # ── 모델 학습 ─────────────────────────────────────────────
    results = model.train(
        data       = DATA_YAML,       # 데이터셋 설정 파일
        epochs     = EPOCHS,          # 총 학습 epoch
        imgsz      = IMG_SIZE,        # 입력 이미지 크기
        batch      = BATCH_SIZE,      # 배치 크기
        workers    = WORKERS,         # 데이터 로딩 워커 수
        device     = device,          # 학습 디바이스
        verbose    = False,           # YOLO 기본 출력 끔 (콜백으로 대체)

        # 학습률 설정
        lr0        = LR0,
        lrf        = LRF,

        patience   = PATIENCE,

        weight_decay = 0.0005,        # 과적합 방지
        warmup_epochs = 5,            # 처음 5 epoch 워밍업

        # 결과 저장 경로
        project    = PROJECT_DIR,
        name       = RUN_NAME,

        # 체크포인트: best.pt 와 last.pt 모두 저장
        save       = True,
        save_period= 10,              # 10 epoch 마다 중간 가중치 저장

        # 학습 모니터링
        plots      = True,            # 학습 곡선, confusion matrix 등 시각화

        # 데이터 증강 설정
        **AUGMENT_CFG,
        ## '**' - 딕셔너리를 풀어서 개별 인자로 전달하는 파이썬 문법
    )

    '''
        weight_decay = 0.0005  - 과적합 방지
    
    #   모델 내부엔 수백만 개의 가중치가 존재
    #   과적합이 생기면 일부 가중치가 비정상적으로 커
    #   너무 크면 ? -> 가중치를 너무 깎아서 학습 자체가 안됨
    
    #   매 epoch마다 모든 가중치에 x 0.9995를 곱해서 강제로 줄임
    '''


    # ── 학습 완료 후 best.pt 경로 출력 ───────────────────────
    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("  학습 완료!")
    print(f"  최고 가중치 저장 위치:\n  {best_weights}")
    print("=" * 55)

    return results


# ============================================================
# 3. valid 함수 (학습 후 성능 확인)
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
    model = YOLO(weights_path)                                 # best.pt
    metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE)        # 검증 실행

    p   = metrics.box.mp            # Precision
    r   = metrics.box.mr            # Recall
    m50 = metrics.box.map50         # mAP50

    print("\n" + "=" * 55)
    print("  최종 검증 결과 (best.pt 기준)")
    print("=" * 55)
    print(f"  Acc       : {m50:.4f}")
    print(f"  Precision : {p:.4f}  탐지한 것 중 {p*100:.1f}% 가 맞음")
    print(f"  Recall    : {r:.4f}  실제 객체의 {r*100:.1f}% 를 찾아냄")
    print("=" * 55)


# ============================================================
# 4. 추론 예시 함수 (단일 이미지 테스트)
# ============================================================

# predict
def predict_sample(image_path: str, weights_path: str = None):
    '''
        학습 완료된 best.pt로 사진 한 장을 보고 '안전벨트 착용/미착용'을 판단하는 함수
    '''    
    if weights_path is None:
        weights_path = str(
            Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
        )

    model = YOLO(weights_path)        # best.pt

    # conf=0.5: 신뢰도 50% 이상인 탐지 결과만 표시
    results = model.predict(
        source  = image_path,
        conf    = 0.5,
        save    = True,               # 결과 이미지 저장
        project = PROJECT_DIR,
        name    = "predict",
    )

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            label  = model.names[cls_id]
            print(f"  감지: {label}  (신뢰도 {conf:.2%})")


# ============================================================
# 5. 진입점
# ============================================================

if __name__ == "__main__":
    train()         # 1. 학습 진행
    # predict_sample(r"C:\경로\테스트이미지.jpg")
