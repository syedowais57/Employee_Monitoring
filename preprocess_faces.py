# preprocess_faces.py

import os
import cv2
import pickle
from utils.face_utils import get_face_locations, encode_faces

# Directory containing face images of known employees
FACE_DB_DIR = 'face_db'
# Output file for encodings
ENCODINGS_FILE = 'encodings.pkl'

def preprocess_faces():
    known_encodings = []
    known_names = []

    # Loop over each image in the face_db directory
    for filename in os.listdir(FACE_DB_DIR):
        filepath = os.path.join(FACE_DB_DIR, filename)
        name, ext = os.path.splitext(filename)
        # Only process image files
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # Read image
        image = cv2.imread(filepath)
        if image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        # Detect faces in the image
        face_locations = get_face_locations(image)
        if len(face_locations) == 0:
            print(f"Warning: No face detected in {filename}. Skipping.")
            continue
        if len(face_locations) > 1:
            print(f"Warning: More than one face detected in {filename}. Using the first face.")

        # Compute face embeddings (encodings)
        encoding = encode_faces(image, [face_locations[0]])  # Only take the first face
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(name)
            print(f"Processed {name}, encoding saved.")
        else:
            print(f"Warning: Could not encode face in {filename}.")

    # Save encodings and names to a pickle file
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nAll encodings saved to {ENCODINGS_FILE}.")

if __name__ == '__main__':
    preprocess_faces()
