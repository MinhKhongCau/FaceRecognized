import tensorflow as tf
from tensorflow.keras import layers, models

# Xây dựng kiến trúc CNN
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)  # 4 giá trị: (x, y, w, h)
    ])
    return model

model = build_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
