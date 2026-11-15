import pandas as pd
import matplotlib.pyplot as plt

# 학습 결과 불러오기
df = pd.read_csv("runs/step3/results.csv")
df.columns = [c.strip() for c in df.columns]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# ① 학습 / 검증 Segmentation loss
for c in ["train/seg_loss", "val/seg_loss"]:
    if c in df.columns:
        ax[0].plot(df["epoch"], df[c], label=c)

ax[0].set_title("Segmentation Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

# ② Mask mAP
for c in ["metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
    if c in df.columns:
        ax[1].plot(df["epoch"], df[c], label=c)

ax[1].set_title("Mask mAP")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("mAP")
ax[1].legend()

plt.tight_layout()
plt.savefig("curve.png", dpi=150)
plt.show()