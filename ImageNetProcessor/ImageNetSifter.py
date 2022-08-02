import os
import argparse
import shutil
import csv
import random
import pickle
from typing import Tuple, List, Dict

import imagenetLabels
from PIL import Image
BASE_DIR = os.environ.get('ImageNetDir')
imagenet_dir = BASE_DIR + '/ImageNetImages224Size'
bird_labels = [
    'cock',
    'hen',
    'ostrich',
    'brambling',
    'goldfinch',
    'house_finch',
    'junco',
    'indigo_bunting',
    'robin',
    'bulbul',
    'jay',
    'magpie',
    'chickadee',
    'water_ouzel',
    'kite',
    'bald_eagle',
    'vulture',
    'great_grey_owl',
    'black_grouse',
    'ptarmigan',
    'ruffed_grouse',
    'prairie_chicken',
    'peacock',
    'quail',
    'partridge',
    'African_grey',
    'macaw',
    'sulphur_crested_cockatoo',
    'lorikeet',
    'coucal',
    'bee_eater',
    'hornbill',
    'hummingbird',
    'jacamar',
    'toucan',
    'drake',
    'red_breasted_merganser',
    'goose',
    'black_swan',
    'white_stork',
    'black_stork',
    'spoonbill',
    'flamingo',
    'American_egret',
    'little_blue_heron',
    'bittern',
    'crane',
    'limpkin',
    'American_coot',
    'bustard',
    'ruddy_turnstone',
    'red_backed_sandpiper',
    'redshank',
    'dowitcher',
    'oystercatcher',
    'European_gallinule',
    'pelican',
    'king_penguin',
    'albatross',
]

def decodeDir(
        directory:str=imagenet_dir
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    bird_directories = set()
    labels = []
    for img_directory in os.listdir(directory):
        underscore_index = img_directory.find('_')
        img_directory_label = img_directory[underscore_index+1:]
        if img_directory_label in bird_labels:
            bird_directories.add(img_directory)
    print(bird_directories)
    for par_dir, _, files in os.walk(directory, followlinks=True):
        is_bird_dir = any((x in par_dir for x in bird_directories))
        labels += [1 if is_bird_dir else 0] * len(files)
        if is_bird_dir:
            print(par_dir, is_bird_dir)
    print(len(labels))
    with open('labels.pk', 'wb') as file:
        pickle.dump(labels, file)
    
    print('Sifted all the data')
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(   
        '-d',
        '--directory',
        help='Directory to decode',
        default=imagenet_dir
    )
    args = parser.parse_args()
    decodeDir(args.directory)