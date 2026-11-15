# 제공된 실습용 데이터셋 확인
import glob
import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

# imgs = sorted(
#     glob.glob("data/images/**/*.jpg", recursive=True)
#     )
# print("영상 수:", len(imgs))
# print("예시:", imgs[0] if imgs else "없음")

def observation(imgs):
    n = min(9, len(imgs))
    sample = random.sample(imgs, n)
    fig, ax = plt.subplots(3, 3, figsize=(10, 9))
    axes = ax.ravel()

    for a in axes:
        a.axis("off")

    for a, p in zip(axes, sample):
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        a.imshow(im)
        a.set_title(Path(p).name[:20], fontsize=7)

    plt.tight_layout()
    plt.savefig("samples9.png", dpi=110)

    # 첫 번째 영상의 밝기 통계 확인
    if imgs:
        gray = cv2.imread(imgs[0], 0)
        print("밝기 평균:", round(gray.mean(), 1))
        print("밝기 표준편차:", round(gray.std(), 1))

if __name__ == "__main__":
    imgs = sorted(
    glob.glob("data/images/**/*.jpg", recursive=True)
    )
    print("영상 수:", len(imgs))
    print("예시:", imgs[0] if imgs else "없음")
    observation(imgs)
