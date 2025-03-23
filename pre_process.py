import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1️⃣ - Định nghĩa mô hình CNN sử dụng Functional API
inputs = tf.keras.Input(shape=(128, 128, 3), name="input_layer")

x = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv1")(inputs)
x = tf.keras.layers.MaxPooling2D(2,2, name="pool1")(x)
x = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2")(x)
x = tf.keras.layers.MaxPooling2D(2,2, name="pool2")(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv3")(x)

model = tf.keras.Model(inputs=inputs, outputs=x, name="FaceDetector")

# ✅ FIX: Tạo mô hình trung gian lấy feature maps
activation_model = tf.keras.Model(inputs=model.input, 
                                  outputs=[layer.output for layer in model.layers if "conv" in layer.name])

# 2️⃣ - Load ảnh và tiền xử lý
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize về 128x128
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển về RGB
    img = img / 255.0  # Chuẩn hóa ảnh về [0,1]
    return np.expand_dims(img, axis=0)  # Thêm batch dimension (1,128,128,3)

# 3️⃣ - Trích xuất đặc trưng từ từng lớp
def get_layer_outputs(image):
    return activation_model.predict(image)

# 4️⃣ - Hiển thị ảnh sau mỗi lớp convolutional
def visualize_feature_maps(image_path):
    image = load_and_preprocess_image(image_path)
    feature_maps = get_layer_outputs(image)

    for i, fmap in enumerate(feature_maps):
        num_filters = fmap.shape[-1]  # Số lượng bộ lọc
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Feature Maps after Conv Layer {i+1}")

        print("number of filter: ", num_filters)

        for j in range(min(num_filters, 16)):  # Hiển thị tối đa 16 bộ lọc
            plt.subplot(2, 8, j+1)
            plt.imshow(fmap[0, :, :, j], cmap="gray")
            plt.axis("off")

        plt.show()

# 5️⃣ - Chạy thử với một ảnh
image_path = "59_peopledrivingcar_peopledrivingcar_59_37.jpg"  # Thay bằng đường dẫn ảnh
visualize_feature_maps(image_path)
