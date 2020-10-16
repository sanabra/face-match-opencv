import face_recognition
import cv2
import numpy as np
from enum import Enum
from load_face import load_face_file


class DetectType(Enum):
    WEBCAM = 'WEBCAM'
    PICTURE = 'PICTURE'

    def __str__(self):
        return self.value


# Display the results
def show(title_window, image, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        # Input text label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        cv2.imshow(title_window, image)


def detect_image(image, faces_encodings, faces_names):
    face_names = []
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) >= 1:
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)
    return face_locations, face_names


def detect_picture_file(picture_file, faces_encodings, faces_names):
    img = load_face_file(picture_file)
    face_locations, face_names = detect_image(img, faces_encodings, faces_names)
    show('Picture', img, face_locations, face_names)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_webcam(faces_encodings, faces_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        frame_image = small_frame[:, :, ::-1]
        face_locations, face_names = detect_image(frame_image, faces_encodings, faces_names)
        show('Video', frame, face_locations, face_names)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
