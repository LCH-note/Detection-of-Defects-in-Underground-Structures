from pathlib import Path
from ultralytics import YOLO

def check_mAP50():
    # 현재 실습 폴더 기준 경로
    BASE_DIR = Path.cwd()

    DATA_YAML = BASE_DIR / "data" / "data.yaml"
    BEST_PT = BASE_DIR / "runs" / "step3" / "weights" / "best.pt"

    # 파일 존재 확인
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {DATA_YAML}")

    if not BEST_PT.exists():
        raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {BEST_PT}")

    # 모델 불러오기
    model = YOLO(str(BEST_PT))

    # 테스트셋 평가
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=960,
        plots=True
        )

    # 전체 성능
    print("mask mAP50 :", round(results.seg.map50, 4))
    print("mask mAP50-95:", round(results.seg.map, 4))
    print("box mAP50 :", round(results.box.map50, 4))

    # 클래스별 mask mAP50-95
    class_names = [
        "crack",
        "efflorescence",
        "spall_detach"
        ]
    for i, name in enumerate(class_names):
        print(
            name,
            "mask mAP50-95:",
            round(results.seg.maps[i], 4)
            )

if __name__ == "__main__":
    check_mAP50()