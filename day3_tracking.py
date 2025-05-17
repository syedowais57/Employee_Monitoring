# day3_tracking.py

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from utils.face_recognition_helper import load_known_faces
from utils.tracking_utils import update_tracks

# 1. Load known face encodings and names
known_encs, known_names = load_known_faces(face_dir="face_db/custom")

# 2. Initialize YOLOv8 for person detection
model = YOLO("yolov8n.pt")

# 3. Open webcam or RTSP stream
cap = cv2.VideoCapture(0)
cv2.namedWindow("Tracking + Recognition", cv2.WINDOW_NORMAL)

# 4. Tracking / recognition state
known_ids = set()
id_to_name = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 5. Detect people
    results = model(frame)[0]
    detections = [
        [int(box.xyxy[0][0]), int(box.xyxy[0][1]),
         int(box.xyxy[0][2]), int(box.xyxy[0][3]),
         float(box.conf[0])]
        for box in results.boxes
        if int(box.cls[0]) == 0
    ]

    # 6. Track and get stable IDs
    tracks = update_tracks(detections, frame=frame)

    # 7. For each tracked box, do face‐based recognition once
    for x1, y1, x2, y2, tid in tracks:
        if tid not in known_ids:
            known_ids.add(tid)

            # Crop the person region and convert to RGB
            crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Detect face locations within that crop
            face_locs = face_recognition.face_locations(rgb_crop, model="hog")
            # Compute encodings for each face found
            encs = face_recognition.face_encodings(rgb_crop, face_locs, num_jitters=1)

            name = "Unknown"
            if encs:
                # Compare first face encoding against known database
                dists = face_recognition.face_distance(known_encs, encs[0])
                best_idx = np.argmin(dists)
                if dists[best_idx] < 0.55:
                    name = known_names[best_idx]

            id_to_name[tid] = name

        # Draw bounding box + label
        label = id_to_name.get(tid, "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}#{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 8. Display and quit
    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Tracking + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
