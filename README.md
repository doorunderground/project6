# In-Cabin Safety Detection System
> 차량 내 탑승자 안전 모니터링 시스템 — 안전벨트 감지 + 졸음운전 경보

---

## 개요

차량 내부 카메라(블랙박스 또는 DMS 카메라)를 활용하여 운전자의 **안전벨트 착용 여부**와 **졸음운전 징후**를 실시간으로 감지하는 YOLO 기반 통합 안전 시스템입니다.

두 단계 파이프라인으로 동작합니다:
- **Phase 1** : 안전벨트 착용을 5초 연속 확인 → 주행 중임을 판단
- **Phase 2** : 얼굴 감지 → 눈 상태 판별 → 2초 이상 눈 감김 시 경보 출력

---

## 프로젝트 구조

```
PROJECT6/
├── main.py               # 통합 실행 (안전벨트 → 졸음 감지 파이프라인)
├── belt_train.py         # 안전벨트 감지 모델 학습 스크립트
├── sleep_train.py        # 졸음 감지(눈 상태) 모델 학습 스크립트
├── belt_simulate.py      # 안전벨트 단독 시뮬레이션 테스트
├── sleep_simulate.py     # 졸음 감지 단독 시뮬레이션 테스트
│
├── models/
│   ├── seatbelt.pt       # 학습된 안전벨트 감지 가중치
│   ├── sleep_detect.pt   # 학습된 눈 상태 감지 가중치 (open / closed)
│   └── yolov8n-face.pt   # 얼굴 감지 모델 (YOLOv8n-face)
│
├── test_video/
│   ├── belt_test1.mp4    # 안전벨트 테스트 영상 1
│   ├── belt_test2.mp4    # 안전벨트 테스트 영상 2
│   ├── sleep_test1.mp4   # 졸음 테스트 영상 1
│   ├── sleep_test2.mp4   # 졸음 테스트 영상 2
│   └── sleep_test3.mp4   # 졸음 테스트 영상 3
│
└── result_video/         # 감지 결과가 오버레이된 출력 영상 저장 폴더
```

---

## 기능 상세

### Phase 1 — 안전벨트 감지 (`seatbelt.pt`)

| 항목 | 내용 |
|------|------|
| 감지 클래스 | `seatbelt` / `no-seatbelt` |
| 활성화 조건 | `seatbelt` 5초 연속 감지 시 Phase 2 전환 |
| 리셋 조건 | `no-seatbelt` 감지 즉시 타이머 초기화 |
| HUD 표시 | 착용 상태 + 진행 프로그레스 바 |

안전벨트 착용이 확인된 이후에만 졸음 감지가 활성화되므로, 주차 중·탑승 전 등 불필요한 오탐을 방지합니다.

---

### Phase 2 — 졸음 운전 감지 (2단계 파이프라인)

전체 프레임에서 직접 눈을 찾으면 배경 물체를 눈으로 오인하는 오탐이 발생합니다.
이를 해결하기 위해 **얼굴 crop 후 눈 감지**하는 2단계 방식을 적용했습니다.

```
전체 프레임
    ↓
[yolov8n-face] 얼굴 감지 + padding 60px
    ├─ 얼굴 없음 → 전체 프레임 fallback
    └─ 얼굴 있음
          ↓
       frame[fy1:fy2, fx1:fx2]  ← 얼굴 영역 crop
          ↓
       [sleep_detect] 눈 상태 판별 (open / closed)
          ↓
       원본 좌표로 변환 → 박스 표시
```

| 항목 | 내용 |
|------|------|
| 감지 클래스 | `open_eye` / `closed_eye` |
| 경보 조건 | `closed_eye` 2초 이상 연속 감지 |
| 경보 표현 | 화면 테두리 빨간색 + `!! WARNING !!` 텍스트 깜빡임 |
| 현재 제한 | 운전자 1인 감지 (가장 신뢰도 높은 얼굴 1개만 처리) |

---

## 모델 학습

### 안전벨트 모델 (`belt_train.py`)

| 항목 | 값 |
|------|----|
| 베이스 모델 | `yolo11s.pt` |
| 학습 이미지 | 약 1,000장 (belt: 600 / no_belt: 400) |
| 검증 이미지 | 약 200장 |
| Epochs | 200 |
| Batch size | 32 |
| 입력 크기 | 640×640 |
| Optimizer | SGD (기본) |
| 학습률 | lr0=0.01 → lrf=0.01 (최종 0.0001) |
| Early stopping | patience=50 |
| 체크포인트 | 10 epoch마다 저장 |

**Augmentation 전략**

| 전략 | 설정값 | 목적 |
|------|--------|------|
| HSV-H / S / V | 1.0 / 1.0 / 0.8 | 낮·밤·터널·흐린 날 조도 변화 대응 |
| degrees / scale / translate | 5.0 / 0.5 / 0.1 | 카메라 왜곡·다양한 자세 대응 |
| fliplr | 0.5 | 좌·우 안전벨트 착용 모두 학습 |
| mosaic / mixup | 1.0 / 0.1 | 가림 현상·경계 상황 학습 |

```bash
python belt_train.py
```

---

### 졸음 감지 모델 (`sleep_train.py`)

| 항목 | 값 |
|------|----|
| 베이스 모델 | 이전 학습 가중치 이어학습 |
| 학습 이미지 | 약 2,700장 |
| 검증 이미지 | 약 650장 |
| Epochs | 200 |
| Batch size | 32 |
| Optimizer | AdamW |
| 학습률 | lr0=0.001 → lrf=0.01 (최종 0.00001) |
| Early stopping | patience=50 |

**Augmentation 전략**

| 전략 | 설정값 | 목적 |
|------|--------|------|
| HSV-V | 0.4 | 낮·밤·터널 조도 변화 대응 |
| degrees | 5.0 | 고개 기울임 각도 대응 |
| fliplr | 0.5 | 좌·우 눈 대칭 학습 |
| flipud | 0.0 | 눈이 뒤집히는 상황 없음 |
| mosaic | 1.0 | 여러 눈 이미지 동시 학습 |

```bash
python sleep_train.py
```

학습 완료 후 자동으로 `validate()` 가 실행되어 Precision / Recall / mAP50 을 출력합니다.

---

## 실행 방법

### 1. 통합 시스템 실행 (권장)

```bash
python main.py
```

`main.py` 상단 설정값으로 소스와 임계값을 조정할 수 있습니다:

```python
VIDEO_SOURCE       = r"C:\PROJECT6\test_video\sleep_test3.mp4"
# VIDEO_SOURCE     = 0   # 웹캠 사용 시

SEATBELT_THRESHOLD = 5.0   # 안전벨트 연속 감지 시간 (초)
ALARM_SECS         = 2     # 눈 감김 경보 기준 시간 (초)
FACE_PAD           = 60    # 얼굴 crop 여유 픽셀
```

| 단축키 | 동작 |
|--------|------|
| `q` / `ESC` | 종료 |
| `Space` | 일시정지 / 재개 |

---

### 2. 안전벨트 단독 시뮬레이션

```bash
# 기본 영상으로 실행
python belt_simulate.py

# 영상·신뢰도 직접 지정
python belt_simulate.py --video test_video/belt_test2.mp4 --conf 0.5
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video` / `-v` | 입력 영상 경로 | `belt_test1.mp4` |
| `--weights` / `-w` | 가중치 파일 경로 | `models/seatbelt.pt` |
| `--conf` / `-c` | 감지 신뢰도 임계값 | `0.4` |
| `--iou` | NMS IoU 임계값 | `0.45` |

종료 후 콘솔에 착용·미착용 프레임 수 및 비율 통계가 출력됩니다.

---

### 3. 졸음 감지 단독 시뮬레이션

```bash
# 기본 영상으로 실행
python sleep_simulate.py

# 소스·FPS·신뢰도 지정
python sleep_simulate.py --source test_video/sleep_test1.mp4 --fps 30 --conf 0.4
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 입력 영상 경로 (미지정 시 `VIDEO_SOURCE`) | `sleep_test1.mp4` |
| `--fps` | 시뮬레이션 FPS | `4` |
| `--conf` | 눈 감지 신뢰도 임계값 | `0.35` |

이미지 폴더를 소스로 지정하면 방향키(`←` / `→`)로 프레임 단위 이동도 가능합니다.

---

## 데이터셋 구조

학습 스크립트 실행 전 아래 구조로 데이터를 준비해야 합니다:

```
PROJECT6/
└── data/
    ├── train/
    │   ├── images/   # 학습 이미지 (.jpg / .png)
    │   └── labels/   # YOLO 형식 라벨 (.txt)
    ├── valid/
    │   ├── images/   # 검증 이미지
    │   └── labels/   # 검증 라벨
    └── data.yaml     # 클래스 정보 및 경로 설정
```

`data.yaml` 예시 (안전벨트):
```yaml
path: C:/PROJECT6/data
train: train/images
val:   valid/images
nc: 2
names: ['no-seatbelt', 'seatbelt']
```

`data.yaml` 예시 (졸음 감지):
```yaml
path: C:/PROJECT6/data
train: train/images
val:   valid/images
nc: 2
names: ['closed_eye', 'open_eye']
```

---

## 의존성

```bash
pip install ultralytics opencv-python torch numpy pyyaml
```

| 패키지 | 용도 |
|--------|------|
| `ultralytics` | YOLOv8 / YOLO11 모델 학습·추론 |
| `opencv-python` | 영상 처리 및 시각화 |
| `torch` | GPU 가속 학습 |
| `numpy` | 배열 연산 |
| `pyyaml` | data.yaml 파싱 |

> GPU 학습을 위해 CUDA 버전에 맞는 PyTorch 설치를 권장합니다.
> CPU 환경에서도 실행되나 학습 속도가 크게 느려집니다.

---

## 주요 파라미터 요약

| 파라미터 | 파일 | 기본값 | 설명 |
|----------|------|--------|------|
| `SEATBELT_THRESHOLD` | `main.py` | `5.0`초 | 졸음 감지 활성화까지 필요한 안전벨트 연속 착용 시간 |
| `ALARM_SECS` | `main.py` | `2`초 | 경보 발생까지 필요한 눈 감김 연속 시간 |
| `SEATBELT_CONF` | `main.py` | `0.4` | 안전벨트 감지 신뢰도 임계값 |
| `FACE_CONF` | `main.py` | `0.4` | 얼굴 감지 신뢰도 임계값 |
| `EYE_CONF` | `main.py` | `0.35` | 눈 상태 감지 신뢰도 임계값 |
| `FACE_PAD` | `main.py` | `60`px | 얼굴 crop 여유 픽셀 (눈썹·이마 포함) |
| `BLINK_INTERVAL` | `main.py` | `0.35`초 | 경고 텍스트 깜빡임 주기 |

---

## 알려진 제한 사항

- 현재 **운전자 1인 단독 감지**만 지원합니다. 다인 감지를 위해서는 안전벨트와 탑승자 간의 매핑 로직이 추가로 필요합니다.
- 마스크 착용·안경·심한 역광 환경에서 얼굴 감지 성능이 저하될 수 있습니다.
- 학습 데이터가 부족한 특정 차종·카메라 각도에서는 오탐이 발생할 수 있습니다.