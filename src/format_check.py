import glob

lbls = sorted(
    glob.glob("YOLO segmentation/train/labels/**/*.txt", recursive=True)
    # glob.glob("data/labels/**/*.txt", recursive=True)
    )

print("라벨 파일 수:", len(lbls))

if not lbls:
    print("라벨 파일이 없습니다.")
else:
    line = open(lbls[0], encoding="utf-8").readline().split()

    print("토큰 개수:", len(line))
    print("앞부분:", line[:9])

    if len(line) == 5:
        print("Detection 포맷입니다.")
    elif len(line) >= 7 and (len(line) - 1) % 2 == 0:
        print("Segmentation 포맷입니다.")
    else:
        print("라벨 형식을 다시 확인하세요.")