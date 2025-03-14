import cv2
import os
import random

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

# Đọc bounding box
for i in range(num_faces):
    x, y, w, h = map(int, lines[idx + 2 + i].strip().split()[:4])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ khung xanh

# Hiển thị ảnh
cv2.imshow("WIDER FACE Sample", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
