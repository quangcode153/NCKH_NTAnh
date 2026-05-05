import numpy as np

path = r"D:\NCKH\MERTools\MER2023\dataset-process\label-6way.npz"
data = np.load(path, allow_pickle=True)

train_dict = data['train_corpus'].item()

print(f"Đã giải nén tập Train! Tổng số video: {len(train_dict)}")
print("-" * 40)
print("🔍 XEM THỬ 3 ĐÁP ÁN ĐẦU TIÊN:")

count = 0
for video_name, label_info in train_dict.items():
    print(f" Video: {video_name}  --->  ĐÁP ÁN: {label_info}")
    count += 1
    if count >= 3:
        break