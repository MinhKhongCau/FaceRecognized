IMG_SIZE = (64, 64)  # Kích thước chuẩn

def load_dataset(annotation_file, image_root):
    print("load image is processing...")
    x_train = []
    y_train = []

    with open(annotation_file, "r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        img_path = os.path.join(image_root, lines[i].strip())  # Đọc đường dẫn ảnh
        num_faces = int(lines[i + 1].strip())  # Số lượng khuôn mặt
        faces = lines[i + 2 : i + 2 + num_faces]  # Danh sách bounding boxes
        print("load image: ", img_path)

        if num_faces == 0:
            num_faces = 1

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            i += 2 + num_faces
            continue  # Bỏ qua nếu ảnh không tồn tại

        h_img, w_img, _ = img.shape
        print(f"Ảnh {img_path} có kích thước: {w_img}x{h_img}")

        for face in faces:
            face_data = list(map(int, face.split()))
            x, y, w, h = face_data[:4]
            label = face_data[4]  # Nhãn của khuôn mặt

            # Cắt khuôn mặt và resize
            face_crop = img[y : y + h, x : x + w]

            if face_crop.size == 0:  # Kiểm tra nếu ảnh bị rỗng
                print("Lỗi: face_crop rỗng, bỏ qua")
                continue
            face_crop = cv2.resize(face_crop, IMG_SIZE)

            x_train.append(face_crop)
            y_train.append(label)

        i += 2 + num_faces

    x_train = np.array(x_train, dtype="float32") / 255.0  # Chuẩn hóa sang dạng [0 -> 1]
    y_train = np.array(y_train)

    return x_train, y_train
