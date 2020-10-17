import argparse
import sys
from load_face import TRAINED_OUTPUT_FILE
from load_face import load_faces_data
from detect import DetectType, detect_picture_file
from detect import detect_webcam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Classification in Python')

    training_facial_arguments = parser.add_argument_group('Training facial arguments')
    training_facial_arguments.add_argument('-d', '--train_dir', nargs='?',
                                           type=str,
                                           help='Directory with images to perform facial recognition training')
    training_facial_arguments.add_argument('-o', '--train_output_file', nargs='?',
                                           type=str, default=TRAINED_OUTPUT_FILE,
                                           help='Output file name with learned data. (default value = {})'
                                           .format(TRAINED_OUTPUT_FILE))

    facial_match_arguments = parser.add_argument_group('Facial match arguments')
    facial_match_arguments.add_argument('-i', '--trained_input_file', nargs='?',
                                        type=str, default=TRAINED_OUTPUT_FILE,
                                        help='Name of the file containing the learned content. '
                                             'Using this parameter optimizes facial recognition. (default value = {})'
                                        .format(TRAINED_OUTPUT_FILE))
    facial_match_arguments.add_argument('-t', '--detect_type',
                                        type=DetectType, choices=list(DetectType), required=True,
                                        help='If you want to test with face recognition, '
                                             'you can use it from an image in the file or via the webcam')

    facial_match_arguments.add_argument('-f', '--file', nargs='?',
                                        type=str,
                                        help='Detect using file')

    args = parser.parse_args()

    faces_encodings, faces_names = load_faces_data(files_dir=args.train_dir,
                                                   output_file=args.train_output_file,
                                                   input_file=args.trained_input_file)

    if args.detect_type == DetectType.WEBCAM:
        detect_webcam(faces_encodings, faces_names)
    else:
        if args.file is None:
            sys.exit('You must inform the name of the file containing the image to be analyzed')
        else:
            detect_picture_file(args.file, faces_encodings, faces_names)
