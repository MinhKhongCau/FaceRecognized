import random
import os
import cv2


# Đường dẫn dataset
dataset_path = "WIDER_train/images"
label_file = "wider_face_train_bbx_gt.txt"

# Đọc file nhãn
with open(label_file, "r") as f:
    lines = f.readlines()

# Lấy ngẫu nhiên 1 ảnh
idx = random.randint(0, len(lines) - 1)
while not lines[idx].strip().endswith(".jpg"):  # Tìm dòng chứa tên file ảnh
    idx += 1

file_name = lines[idx].strip()
num_faces = int(lines[idx + 1].strip())


# Đọc ảnh
img_path = os.path.join(dataset_path, file_name)
image = cv2.imread(img_path)

print(image.shape)