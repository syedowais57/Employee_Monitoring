import os, cv2, face_recognition

BASE_DIR = "face_db/custom"
for person in os.listdir(BASE_DIR):
    in_dir  = os.path.join(BASE_DIR, person)
    out_dir = os.path.join(BASE_DIR, person + "_faces")
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(in_dir):
        img_path = os.path.join(in_dir, img_name)
        image = cv2.imread(img_path)
        # detect face locations
        locations = face_recognition.face_locations(image)
        for i, (top, right, bottom, left) in enumerate(locations):
            face = image[top:bottom, left:right]
            cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}_f{i}.jpg"), face)
