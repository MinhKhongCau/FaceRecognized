import cv2
import os
import numpy as np

# Đọc dữ liệu từ thư mục chứa ảnh
data_dir = "WIDER_train/images"
labels_file = "wider_face_train_bbx_gt.txt"

# Hàm đọc ảnh và box
def load_data():
    images = []
    labels = []
    with open(labels_file, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            file_name = lines[i].strip()
            num_faces = int(lines[i+1].strip())
            img = cv2.imread(os.path.join(data_dir, file_name))
            # for j in range(num_faces):
            #     x, y, w, h = map(int, lines[i+2+j].strip().split()[:4])
            #     images.append(cv2.resize(img, (128, 128)))
            #     labels.append([x, y, w, h])
            # i += num_faces + 2
    return np.array(images) / 255.0, np.array(labels)

X, y = load_data()
