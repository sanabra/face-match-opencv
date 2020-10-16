from train import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Face Classification in Python')
    parser.add_argument('--traindir', nargs='?',
                        const='./ resouces',
                        type=str,
                        default='C:/Users/sanab/Dev/repo-git/face-match-opencv/resources',
                        help='Images diretory to train')

    args = parser.parse_args()
    train(args.traindir)
