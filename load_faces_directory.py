import face_recognition
import os


# ARTICLE https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340
def load_faces_directory(files_dir):
    encodings = []
    names = []

    if files_dir[-1] != '/':
        files_dir += '/'
    train_dir = os.listdir(files_dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(files_dir + person)

        if len(pix) == 0:
            print("rm -R " + files_dir + person)
        else:
            # Loop through each training image for the current person
            for person_img in pix:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(files_dir + person + "/" + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)

                # If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Add face encoding for current image
                    # with corresponding label (name) to the training data
                    encodings.append(face_enc)
                    names.append(person)
                else:
                    print(person + "/" + person_img + " can't be used for training")
                    os.remove(files_dir + person + "/" + person_img)

    result = {
        'names': names,
        'encodings': encodings
    }
    return result
