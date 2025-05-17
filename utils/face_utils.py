# utils/face_utils.py

import cv2
import mediapipe as mp
import face_recognition

mp_face_detection = mp.solutions.face_detection

def get_face_locations(image, min_confidence=0.5):
    """
    Detect faces in the image using MediaPipe.
    Returns a list of (top, right, bottom, left) tuples for each face.
    """
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Initialize MediaPipe FaceDetection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_confidence) as face_detector:
        results = face_detector.process(rgb_image)
    face_locations = []
    img_height, img_width = rgb_image.shape[:2]

    if results.detections:
        for detection in results.detections:
            # Get relative bounding box and convert to absolute pixel values
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * img_width)
            y = int(bbox.ymin * img_height)
            w = int(bbox.width * img_width)
            h = int(bbox.height * img_height)
            # Compute (top, right, bottom, left)
            top = max(y, 0)
            left = max(x, 0)
            bottom = min(y + h, img_height)
            right = min(x + w, img_width)
            face_locations.append((top, right, bottom, left))
    return face_locations

def encode_faces(image, face_locations):
    """
    Compute 128D face encodings for each face found at the given locations.
    Returns a list of encodings.
    """
    # Convert to RGB for face_recognition
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    return encodings

def draw_labels(image, face_locations, names):
    """
    Draw rectangles and name labels on the image for each face.
    """
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label background
        cv2.rectangle(image, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        # Draw label text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 2, bottom - 5), font, 0.5, (0, 0, 0), 1)
