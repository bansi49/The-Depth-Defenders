import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found!")
        break

    # ONLY cell phone = class 67
    results = model.predict(
        source=frame,
        conf=0.25,
        classes=[67],
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
