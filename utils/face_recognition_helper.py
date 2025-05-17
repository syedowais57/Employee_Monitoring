# utils/face_recognition_helper.py

import os
import pickle
import face_recognition

def load_known_faces(face_dir="face_db/custom"):
    """
    Scan subfolders of face_dir, encode each image, and return lists of
    embeddings and corresponding names.
    """
    known_encs = []
    known_names = []

    for person in os.listdir(face_dir):
        person_path = os.path.join(face_dir, person)
        if not os.path.isdir(person_path):
            continue
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            try:
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image, num_jitters=1)
                if encs:
                    known_encs.append(encs[0])
                    known_names.append(person)
            except Exception:
                # skip unreadable images
                continue

    return known_encs, known_names

def save_encodings(encodings, names, out_path="face_db/encodings.pkl"):
    """Serialize known face embeddings and names to a file."""
    with open(out_path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
