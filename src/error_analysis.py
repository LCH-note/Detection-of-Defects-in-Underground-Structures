from ultralytics import YOLO
from pathlib import Path
import cv2


class_names = [
"crack",
"efflorescence",
"spall_detach"
]

SAVE_DIR = Path("error_cases")
SAVE_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGE_DIR = BASE_DIR / "data" / "images" / "test"
TEST_LABEL_DIR = BASE_DIR / "data" / "labels" / "test"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def error_analysis(model, test_imgs, test_gts, best):

    for cid, name in enumerate(class_names):
        fp_imgs = []
        fn_imgs = []
        
        for p, gt in zip(test_imgs, test_gts):
            r = model.predict(
            p,
            imgsz=960,
            conf=0.001,
            verbose=False
            )[0]
            
            conf = r.boxes.conf.cpu().numpy()
            pred_cls = r.boxes.cls.cpu().numpy().astype(int)
            
            # 검증셋에서 정한 클래스별 임계값 사용
            pred_on = bool(
            ((pred_cls == cid) & (conf >= best[name])).any()
            )
            
            gt_on = cid in gt
            
            if pred_on and not gt_on:
                fp_imgs.append(p)
           
            if not pred_on and gt_on:
                fn_imgs.append(p)
            
            print(f"{name} | FP={len(fp_imgs)} | FN={len(fn_imgs)}")
            
            # 클래스별 저장 폴더
            class_dir = SAVE_DIR / name
            class_dir.mkdir(exist_ok=True)
            
            # 대표 FP/FN 최대 3장씩 저장
            for error_type, paths in [
                ("FP", fp_imgs),
                ("FN", fn_imgs)
                ]:
            
                for i, p in enumerate(paths[:3]):
                    r = model.predict(
                        p,
                        imgsz=960,
                        verbose=False
                        )[0]
                    
                    out_path = (
                        class_dir /
                        f"{error_type}_{i+1}_{Path(p).name}"
                        )
                    
                    cv2.imwrite(
                        str(out_path),
                        r.plot()
                    )

if __name__ == "__main__":
    model_path = BASE_DIR / "runs" / "step3" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    # 테스트 이미지 경로
    test_imgs = sorted(
        str(path) for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    # 각 테스트 이미지와 동일한 이름의 라벨에서 클래스 ID 추출
    test_gts = []

    for image_path in test_imgs:
        label_path = TEST_LABEL_DIR / f"{Path(image_path).stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines()

        test_gts.append([
            int(line.split()[0])
            for line in lines
            if line.strip()
            ])

    # 검증셋에서 선택한 임계값
    best = {
        "crack": 0.128,
        "efflorescence": 0.017,
        "spall_detach": 0.026,
        }

    error_analysis(
        model,
        test_imgs,
        test_gts,
        best,
        )
    