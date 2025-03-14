ef preprocess_image(image, label):
#     # Resize ảnh về kích thước chuẩn
#     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
#     # Chuyển ảnh về thang xám
#     image = tf.image.rgb_to_grayscale(image)
#     # Chuẩn hóa ảnh về [0,1]
#     image = image / 255.0
#     return image, label

# # # # Áp dụng tiền xử lý cho toàn bộ dataset
# dataset