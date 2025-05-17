# build_encodings.py
import os, pickle, face_recognition

FACE_DIR = "face_db/custom_faces"
encodings = []
names     = []

for folder in os.listdir(FACE_DIR):
    person = folder.replace("_faces", "")
    path   = os.path.join(FACE_DIR, folder)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = face_recognition.load_image_file(img_path)
        encs  = face_recognition.face_encodings(image)
        if encs:
            encodings.append(encs[0])
            names.append(person)

with open("face_db/encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)
print(f"Saved {len(encodings)} face encodings.")
