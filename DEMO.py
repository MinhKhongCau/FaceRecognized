cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize ảnh và dự đoán gương mặt
    img_resized = cv2.resize(frame, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    pred = model.predict(img_input)[0]
    x, y, w, h = map(int, pred)

    # Vẽ khung quanh gương mặt
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
