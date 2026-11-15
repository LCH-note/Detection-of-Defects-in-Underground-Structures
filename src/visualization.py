import random
import shutil
from pathlib import Path
import yaml
# =========================
# 경로 설정
# =========================
SRC_IMG_DIR = Path("dataset/images")
SRC_LBL_DIR = Path("dataset/labels")
OUT_ROOT = Path("dataset_split")
# =========================
# 설정
# =========================
SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)

CLASS_NAMES = {
0: "crack",
1: "efflorescence",
2: "spall_detach"
}

random.seed(SEED)

# =========================
# 원본 ID 추출 함수
# 파일명 규칙에 맞게 수정
# 예: abc_aug1.jpg -> abc
# =========================
def get_origin_id(stem):
    return stem.split("_aug")[0]

# =========================
# 이미지 목록 확인
# =========================
images = sorted(SRC_IMG_DIR.glob("*.jpg"))
if not images:
    raise FileNotFoundError("data/images 폴더에 jpg 이미지가 없습니다.")

# 원본 ID별 그룹 생성
groups = {}
for img_path in images:
    origin_id = get_origin_id(img_path.stem)
    
    if origin_id not in groups:
        groups[origin_id] = []
    
    groups[origin_id].append(img_path)

origin_ids = list(groups.keys())
random.shuffle(origin_ids)

# =========================
# train / val / test 분할
# =========================

n = len(origin_ids)

n_train = int(n * SPLIT_RATIO[0])
n_val = int(n * SPLIT_RATIO[1])

train_ids = set(origin_ids[:n_train])
val_ids = set(origin_ids[n_train:n_train + n_val])
test_ids = set(origin_ids[n_train + n_val:])

split_ids = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
    }

print("원본 ID 수")
print("train:", len(train_ids))
print("val :", len(val_ids))
print("test :", len(test_ids))

# =========================
# 출력 폴더 생성
# =========================
for split in ["train", "val", "test"]:
    (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# =========================
# 이미지 / 라벨 복사
# =========================
for split, ids in split_ids.items():
    for origin_id in ids:
        for img_path in groups[origin_id]:
            
            # 이미지 복사
            dst_img = OUT_ROOT / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 대응 라벨
            lbl_path = SRC_LBL_DIR / f"{img_path.stem}.txt"
            
            if lbl_path.exists():
                dst_lbl = OUT_ROOT / "labels" / split / lbl_path.name
                shutil.copy2(lbl_path, dst_lbl)
            else:
                print(f"[확인 필요] 라벨 없음: {img_path.name}")

# =========================
# data.yaml 생성
# =========================

data_yaml = {
    "path": str(OUT_ROOT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": CLASS_NAMES
    }

with open(OUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(
        data_yaml,
        f,
        allow_unicode=True,
        sort_keys=False
        )
print("\n분할 완료")
print("data.yaml:", OUT_ROOT / "data.yaml")