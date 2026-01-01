import face_recognition
import pickle
import numpy as np

# Load stored encodings
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_ids = pickle.load(f)

# Load input image
input_image_path = "input/test2.jpeg"
input_image = face_recognition.load_image_file(input_image_path)
input_encodings = face_recognition.face_encodings(input_image)

if len(input_encodings) == 0:
    print("❌ No face detected in input image")
    exit()

input_encoding = input_encodings[0]

# Compare faces
distances = face_recognition.face_distance(known_encodings, input_encoding)
best_match_index = np.argmin(distances)
min_distance = distances[best_match_index]

THRESHOLD = 0.6  # standard threshold

if min_distance < THRESHOLD:
    print("✅ Match Found")
    print("Matched Image ID:", known_ids[best_match_index])
else:
    print("❌ No Match Found")
