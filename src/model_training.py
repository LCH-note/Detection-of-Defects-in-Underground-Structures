from ultralytics import YOLO

def main():
    # 사전학습된 YOLOv8s-Seg 모델 불러오기
    model = YOLO("YOLO model/yolov8s-seg.pt")

    # 모델 학습
    model.train(
        data="data/data.yaml", # 데이터셋 설정 파일
        epochs=60, # 전체 학습 횟수
        imgsz=960, # 입력 이미지 크기
        batch=4, # GPU 메모리 부족 시 2로 조정
        optimizer="AdamW", # 옵티마이저
        lr0=0.00015, # 초기 학습률
        mosaic=0.0, # Mosaic 증강 비활성화
        mixup=0.0, # MixUp 증강 비활성화
        patience=20, # Early Stopping 기준
        seed=42, # 재현성을 위한 시드 고정
        project="runs",
        name="step3"
        )

if __name__ == "__main__":
    main()