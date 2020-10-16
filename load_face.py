import face_recognition
import os
import pickle
import sys

TRAINED_OUTPUT_FILE = 'data.dat'


def save_trained_file(result, output_file):
    file_handler = open(output_file, 'wb')
    pickle.dump(result, file_handler)


def load_trained_file(trained_file):
    file_handler = open(trained_file, 'rb')
    return pickle.load(file_handler)


def load_face_file(face_file):
    if os.path.exists(face_file):
        return face_recognition.load_image_file(face_file)
    else:
        sys.exit('Could not load image. File {} not found.'.format(face_file))


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
                face = load_face_file(files_dir + person + "/" + person_img)
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

    return {'encodings': encodings, 'names': names}


def load_faces(files_dir=None, output_file=TRAINED_OUTPUT_FILE, input_file=TRAINED_OUTPUT_FILE):
    if files_dir is not None:
        result = load_faces_directory(files_dir)
        if output_file is not None:
            save_trained_file(result, output_file)
    else:
        if os.path.exists(input_file):
            result = load_trained_file(input_file)
        else:
            sys.exit('File containing the learned content not exists. File {} not found.'.format(input_file))

    return result['encodings'], result['names']
