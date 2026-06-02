from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

CELL_PHONE_CLASS_ID = 67

# Try camera indexes 0, 1, 2
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# Check if camera opened
if not cap.isOpened():
    print(f"Camera {camera_index} failed. Trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera 1 failed. Trying index 2...")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("ERROR: No camera found! Check your webcam connection.")
    exit()

print("Camera opened successfully! Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame. Exiting...")
        break

    # Detect only cell phones
    results = model(frame, classes=[CELL_PHONE_CLASS_ID])

    annotated_frame = results[0].plot()

    cv2.imshow("Cell Phone Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()