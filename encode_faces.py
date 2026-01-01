import face_recognition
import os
import pickle

DATABASE_DIR = "database"

known_encodings = []
known_ids = []

for file_name in os.listdir(DATABASE_DIR):
    if file_name.lower().endswith((".jpg",".jpeg", ".png")):
        image_path = os.path.join(DATABASE_DIR, file_name)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No face found in {file_name}")
            continue

        known_encodings.append(encodings[0])
        image_id = os.path.splitext(file_name)[0]
        known_ids.append(image_id)

print("Total faces encoded:", len(known_encodings))

with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_ids), f)

print("✅ face_encodings.pkl created successfully")
