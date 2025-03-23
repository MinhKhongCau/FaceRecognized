import cv2
from mtcnn import MTCNN

# Khởi tạo webcam và MTCNN detector
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (nếu có nhiều camera, thử cap = cv2.VideoCapture(1))
detector = MTCNN()

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển ảnh từ BGR sang RGB (MTCNN yêu cầu RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện gương mặt
    faces = detector.detect_faces(frame_rgb)

    # Vẽ khung xung quanh gương mặt
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Khung màu xanh lá
        
        # Hiển thị điểm đặc trưng (mắt, mũi, miệng)
        for key, point in face["keypoints"].items():
            cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Điểm màu đỏ
    
    # Hiển thị kết quả
    cv2.imshow("Face Detection", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
