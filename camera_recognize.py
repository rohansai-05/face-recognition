import cv2
import face_recognition
import pickle
import numpy as np

# Load face encodings
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_ids = pickle.load(f)

# Start webcam
video_capture = cv2.VideoCapture(0)

THRESHOLD = 0.6

print("📷 Camera started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]

        if min_distance < THRESHOLD:
            name = known_ids[best_match_index]
            color = (0, 255, 0)  # Green
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Recognition - Camera", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
