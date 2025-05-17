

import cv2
import os
import pickle
import time
import datetime
import csv
from utils.face_utils import get_face_locations, encode_faces, draw_labels
import face_recognition

# Load face encodings and names
ENCODINGS_FILE = 'face_db/encodings.pkl'
with open(ENCODINGS_FILE, 'rb') as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Prepare logs directory and file
LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'attendance_log.csv')
os.makedirs(LOG_DIR, exist_ok=True)
# If log file doesn't exist, write header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name"])

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting video stream. Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame to improve speed (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Get face locations on the smaller frame
    face_locations = get_face_locations(small_frame)

    names_in_frame = []
    # Convert face_locations back to original frame scale (since we resized)
    scaled_locations = []
    for (top, right, bottom, left) in face_locations:
        top *= 2; right *= 2; bottom *= 2; left *= 2
        scaled_locations.append((top, right, bottom, left))

    # Encode faces on the original (or resized) frame
    encodings = encode_faces(frame, scaled_locations)

    # Compare encodings to known faces
    for encoding, location in zip(encodings, scaled_locations):
        matches = []
        name = "Unknown"
        if known_encodings:
            # Compute distances to known faces
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = distances.argmin() if len(distances) > 0 else None
            if best_match_index is not None and distances[best_match_index] < 0.6:
                name = known_names[best_match_index]
        names_in_frame.append(name)

        # Log the attendance (timestamp and name)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name])

    # Draw boxes and labels
    draw_labels(frame, scaled_locations, names_in_frame)

    # Display the resulting frame
    cv2.imshow('Employee Monitoring', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
