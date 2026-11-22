import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "runs" / "step3" / "weights" / "best.pt"
MODEL_NANO_PATH = BASE_DIR / "runs" / "step3n" / "weights" / "best.pt"
MODEL_MEDIUM_PATH = BASE_DIR / "runs" / "step3m" / "weights" / "best.pt"

VAL_IMAGE_DIR = BASE_DIR / "data" / "images" / "val"
VAL_LABEL_DIR = BASE_DIR / "data" / "labels" / "val"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_IMAGE_DIR = BASE_DIR / "data" / "images" / "test"
TEST_LABEL_DIR = BASE_DIR / "data" / "labels" / "test"


def onoff(model, paths, gts, thr, cid_target):
    tp = fp = fn = tn = 0
    
    for p, gt in zip(paths, gts):
        # 예측 수행
        r = model.predict(
            p,
            imgsz=960,
            conf=0.001,
            verbose=False
            )[0]
    
        # 예측 클래스와 confidence
        conf = r.boxes.conf.cpu().numpy()
        cid = r.boxes.cls.cpu().numpy().astype(int)

        # 해당 클래스의 예측 중 threshold 이상이 하나라도 있으면 ON
        pred_on = bool(
            ((cid == cid_target) & (conf >= thr)).any()
            )
       
        # 정답 라벨에 해당 클래스가 있으면 ON
        gt_on = cid_target in gt

        # TP / FP / FN / TN 계산
        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
        )
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1
        }

def explore(path):
    # THRS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]
    THRS = np.arange(0.001, 1.0, 0.001)
    
    class_names = [
        "crack",
        "efflorescence",
        "spall_detach"
        ]
    
    best = {}
    all_scores = {}

    model = YOLO(str(path))

    # 임계값 탐색에 사용할 검증셋 이미지 경로
    paths = sorted(
        str(path) for path in VAL_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    # 각 이미지와 이름이 같은 YOLO 라벨에서 정답 클래스들을 읽는다.
    gts = []
    for image_path in paths:
        label_path = VAL_LABEL_DIR / f"{Path(image_path).stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines()
        gts.append([int(line.split()[0]) for line in lines if line.strip()])

    for cid, name in enumerate(class_names):
        scores = []

        for t in THRS:
            r = onoff(
                model,
                paths,
                gts,
                t,
                cid
                )
            scores.append(r["f1"])
        
            print(
                f"{name} | thr={t:.3f} | "
                f"F1={r['f1']:.3f} | "
                f"FP={r['fp']} | FN={r['fn']}"
                )
        
        all_scores[name] = scores
        
        # 최고 F1
        max_f1 = max(scores)
        
        # 동점이면 더 낮은 threshold 선택
        best_idx = next(
        i for i, s in enumerate(scores)
        if s == max_f1
        )
        
        best[name] = THRS[best_idx]
        
        print(
            f">> {name} 선택 임계값: "
            f"{THRS[best_idx]} "
            f"(F1={max_f1:.3f})\n"
            )
    
    print("최종 임계값:", best)
    
    # F1 곡선 시각화
    for name in class_names:
    #     plt.plot(
    #         THRS,
    #         all_scores[name],
    #         marker="o",
    #         label=name
    #         )
        scores = all_scores[name]

        # 전체 결과는 선으로만 표시
        line, = plt.plot(
            THRS,
            scores,
            label=name
            )

        # 선택된 임계값과 가장 가까운 위치
        best_idx = int(np.argmin(np.abs(THRS - best[name])))
        best_thr = THRS[best_idx]
        best_f1 = scores[best_idx]

        # 선택된 임계값만 점으로 표시
        plt.scatter(
            best_thr,
            best_f1,
            color=line.get_color(),
            s=80,
            edgecolors="black",
            zorder=3
            )
        plt.annotate(
        f"{best_thr:.3f}",
        (best_thr, best_f1),
        xytext=(5, 8),
        textcoords="offset points",
        color=line.get_color()
        )
    
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1-score")
    plt.title("Validation F1 by Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("val_threshold_f1_medium.png", dpi=150)
    plt.show()


def pick_best_thr(best, path):
    class_names = [
        "crack",
        "efflorescence",
        "spall_detach"
        ]

    model = YOLO(str(path))

    # 임계값 탐색에 사용할 검증셋 이미지 경로
    test_imgs = sorted(
        str(path) for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    # 각 이미지와 이름이 같은 YOLO 라벨에서 정답 클래스들을 읽는다.
    test_gts = []
    for image_path in test_imgs:
        label_path = TEST_LABEL_DIR / f"{Path(image_path).stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines()
        test_gts.append([int(line.split()[0]) for line in lines if line.strip()])
    
    rows = []
    for cid, name in enumerate(class_names):
        r = onoff(
            model,
            test_imgs,
            test_gts,
            best[name], # 검증셋에서 선택한 임계값 그대로 사용
            cid
            )
        rows.append([
            name,
            best[name],
            r["tp"],
            r["fp"],
            r["fn"],
            r["tn"],
            r["precision"],
            r["recall"],
            r["f1"]
            ])
    df = pd.DataFrame(
        rows,
        columns=[
            "class",
            "threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "precision",
            "recall",
            "f1"
            ]
        )
    print(df.round(3))
    print("macro F1:", round(df["f1"].mean(), 3))

    

if __name__ == "__main__":
    # explore(MODEL_MEDIUM_PATH)

    # YOLOv8s 모델
    # pick_best_thr({
    #     "crack": 0.128,
    #     "efflorescence": 0.017,
    #     "spall_detach": 0.026
    #     }, MODEL_PATH)

    # YOLOv8n 모델
    pick_best_thr({
        "crack": 0.3,
        "efflorescence": 0.011,
        "spall_detach": 0.026
        }, MODEL_NANO_PATH)
    
    # YOLOv8m 모델
    pick_best_thr({
        "crack": 0.253,
        "efflorescence": 0.325,
        "spall_detach": 0.131
        }, MODEL_MEDIUM_PATH)
    