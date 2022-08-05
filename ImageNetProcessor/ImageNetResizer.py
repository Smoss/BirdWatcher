import png
import os
# import pickle
import numpy
import tarfile
import argparse
import tensorflow as tf
from imagenetLabels import imagenet_labels, imagenet_original_labels, skip_labels
from PIL import Image
from PIL.Image import Resampling
from ImageNetSifter import bird_labels

# imagenet_dir = './ILSVRC2012_img_train'
BASE_DIR = os.environ.get('ImageNetDir')
INITIAL_DIR = BASE_DIR + '/ImageNetImagesUnsized'
TARGET_SIZE = 380
TARGET_DIR = BASE_DIR + f'''/ImageNetImages{TARGET_SIZE}Size'''

def decodeDir(directory:str=INITIAL_DIR, target_dir:str=TARGET_DIR, target_size:int=TARGET_SIZE) -> None:
    for label in os.listdir(directory):
        from_dir = '{}/{}'.format(directory, label)
        sub_dir = 'bird'
        inc = 1
        if label not in bird_labels:
            sub_dir = 'not_bird'
            inc = 10
        else:
            print(f'It\'s a bird label {label}')

        to_dir = '{}/{}/{}'.format(target_dir, sub_dir, label)
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        for file in os.listdir(from_dir)[::inc]:
            from_path = '{}/{}'.format(from_dir, file)
            target_path = '{}/{}'.format(to_dir, file)
            im = Image.open(from_path)
            target_im = im.resize((target_size, target_size), resample=Resampling.BICUBIC)
            try:
                target_im.save(target_path)
            except:
                print(target_path)
        print('Finished handling', to_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(
        '-d',
        '--directory',
        help='Directory to decode',
        default=INITIAL_DIR
    )
    parser.add_argument(
        '-t',
        '--target',
        help='Directory to decode',
        default=TARGET_DIR
    )
    parser.add_argument(
        '-s',
        '--size',
        help='Desired pixel size of the output images',
        default=TARGET_SIZE,
        type=int
    )
    args = parser.parse_args()
    decodeDir(args.directory, args.target, args.size)