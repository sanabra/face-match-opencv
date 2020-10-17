import multiprocessing
from functools import partial
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


def load_file(face_file):
    if os.path.exists(face_file):
        return face_recognition.load_image_file(face_file)
    else:
        sys.exit('Could not load image. File {} not found.'.format(face_file))


def load_person_face(person_img, files_dir, person):

    # Get the face encodings for the face in each image file
    face = load_file(files_dir + person + "/" + person_img)
    face_bounding_boxes = face_recognition.face_locations(face)

    # If training image contains exactly one face
    if len(face_bounding_boxes) == 1:
        return face_recognition.face_encodings(face)[0]
    else:
        print(person + "/" + person_img + " can't be used for training")
        os.remove(files_dir + person + "/" + person_img)
        return


def load_person_directory(person, files_dir, encodings, names):
    print('Loading {} faces'.format(person))
    pix = os.listdir(files_dir + person)

    if len(pix) == 0:
        print("rm -R " + files_dir + person)
    else:
        for person_img in pix:
            encoding = load_person_face(person_img, files_dir, person)
            if encoding is not None:
                encodings.append(encoding)
                names.append(person)

    return encodings, names


def load_person_directory_paralel(person, files_dir):
    print('Loading {} faces'.format(person))
    pix = os.listdir(files_dir + person)

    if len(pix) == 0:
        print("rm -R " + files_dir + person)
        return [], []
    else:
        encodings = []
        names = []
        for person_img in pix:
            encoding = load_person_face(person_img, files_dir, person)
            if encoding is not None:
                encodings.append(encoding)
                names.append(person)

        return encodings, names


def load_faces_directory(files_dir):

    if files_dir[-1] != '/':
        files_dir += '/'
    train_dir = os.listdir(files_dir)

    pool = multiprocessing.Pool(20)
    _load_person_directory = partial(load_person_directory_paralel, files_dir=files_dir)
    encodings, names = zip(*pool.map(_load_person_directory, train_dir))

    encodings = [item for sublist in encodings for item in sublist]
    names = [item for sublist in names for item in sublist]

    return {'encodings': encodings, 'names': names}


def load_faces_data(files_dir=None, output_file=TRAINED_OUTPUT_FILE, input_file=TRAINED_OUTPUT_FILE):
    if files_dir is not None:
        result = load_faces_directory(files_dir)
        if output_file is not None:
            save_trained_file(result, output_file)
    else:
        if os.path.exists(input_file):
            result = load_trained_file(input_file)
        else:
            sys.exit('File containing the learned content not exists. File {} not found.'.format(input_file))
    print('Total people = {}'.format(len(set(result['names']))))
    print('Total faces = {}'.format(len(result['encodings'])))
    return result['encodings'], result['names']
