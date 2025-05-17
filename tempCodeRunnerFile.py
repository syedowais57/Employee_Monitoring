from ultralytics import YOLO
import cv2

# --- Load YOLOv8 pretrained model ---
model = YOLO("yolov8n.pt")  # lightweight model

# --- Use webcam (use RTSP URL or video file if needed) ---
source = "rtsp://saibbyweb:9906697711Sw#192.168.1.14:554/stream1"  # change this to your CCTV RTSP stream if needed

cap = cv2.VideoCapture(source)

# Check if camera opened successfully
if not cap.isOpened():
    print("[ERROR] Cannot open video source")
    exit()

# Create a named window to capture key presses
cv2.namedWindow("YOLOv8 People Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # --- Run detection ---
    results = model(frame)[0]  # first result (YOLOv8 returns a list)

    # --- Annotate frame with results ---
    annotated_frame = results.plot()

    # --- Add "Press Q to quit" message ---
    cv2.putText(
        annotated_frame,
        "Press 'q' to quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # --- Display the frame ---
    cv2.imshow("YOLOv8 People Detection", annotated_frame)

    # --- Quit if 'q' is pressed ---
    key = cv2.waitKey(1)
    if key == ord("q"):
        print("[INFO] Quitting...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
