import glob
from pathlib import Path

bad = set()

for p in glob.glob(
    # 'data/labels/**/*.txt', recursive=True
    'YOLO segmentation/train/labels/**/*.txt', recursive=True
                   ):
    for ln in open(p):
        v = ln.split()
        if not v:
            continue
        if int(v[0]) not in (0, 1, 2):
            bad.add(p)
        if any(not 0 <= float(x) <= 1 for x in v[1:]):
            bad.add(p)

print('이상 파일:', len(bad))

