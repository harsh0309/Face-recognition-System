import cv2
import os

# Set your name
person_name = "Nitin"
save_dir = os.path.join("dataset", person_name)

# Create folder if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("📸 Press 's' to save a photo, 'q' to quit")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture Face - Press 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face_img)
            print(f"✅ Saved image {count}.jpg")
            count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
