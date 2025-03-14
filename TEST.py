def detect_face(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    
    pred = model.predict(img_input)[0]  # Dự đoán tọa độ (x, y, w, h)
    x, y, w, h = map(int, pred)

    # Vẽ khung xung quanh gương mặt
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test trên ảnh
detect_face("test_image.jpg")
