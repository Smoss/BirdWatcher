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
IMAGE_SIZE = 380
imagenet_dir = BASE_DIR + f'/ImageNetImages{IMAGE_SIZE}Size'
bird_labels = [
    "n01514668_cock",
    "n01514859_hen",
    "n01518878_ostrich",
    "n01530575_brambling",
    "n01531178_goldfinch",
    "n01532829_house_finch",
    "n01534433_junco",
    "n01537544_indigo_bunting",
    "n01558993_robin",
    "n01560419_bulbul",
    "n01580077_jay",
    "n01582220_magpie",
    "n01592084_chickadee",
    "n01601694_water_ouzel",
    "n01608432_kite",
    "n01614925_bald_eagle",
    "n01616318_vulture",
    "n01622779_great_grey_owl",
    "n01795545_black_grouse",
    "n01796340_ptarmigan",
    "n01797886_ruffed_grouse",
    "n01798484_prairie_chicken",
    "n01806143_peacock",
    "n01806567_quail",
    "n01807496_partridge",
    "n01817953_African_grey",
    "n01818515_macaw",
    "n01819313_sulphur_crested_cockatoo",
    "n01820546_lorikeet",
    "n01824575_coucal",
    "n01828970_bee_eater",
    "n01829413_hornbill",
    "n01833805_hummingbird",
    "n01843065_jacamar",
    "n01843383_toucan",
    "n01847000_drake",
    "n01855032_red_breasted_merganser",
    "n01855672_goose",
    "n01860187_black_swan",
    "n02002556_white_stork",
    "n02002724_black_stork",
    "n02006656_spoonbill",
    "n02007558_flamingo",
    "n02009229_little_blue_heron",
    "n02009912_American_egret",
    "n02011460_bittern",
    "n02012849_crane",
    "n02013706_limpkin",
    "n02017213_European_gallinule",
    "n02018207_American_coot",
    "n02018795_bustard",
    "n02025239_ruddy_turnstone",
    "n02027492_red_backed_sandpiper",
    "n02028035_redshank",
    "n02033041_dowitcher",
    "n02037110_oystercatcher",
    "n02051845_pelican",
    "n02056570_king_penguin",
    "n02058221_albatross",
]

def decodeDir(
        directory:str=imagenet_dir
    ) -> None:
    bird_directories = set()
    labels = []
    checked_labels = set()
    num_birds = 0
    num_not_birds = 0
    for img_directory in os.listdir(directory):
        # underscore_index = img_directory.find('_')
        # img_directory_label = img_directory[underscore_index+1:]
        if img_directory in checked_labels:
            print(img_directory)
        checked_labels.add(img_directory)
        if img_directory in bird_labels:
            bird_directories.add(img_directory)
    # print(set(bird_labels))
    print(len(bird_directories) - len(bird_labels))
    for par_dir, _, files in os.walk(directory, followlinks=True):
        # print(par_dir, files)
        is_bird_dir = '\\bird\\' in par_dir
        labels += [1 if is_bird_dir else 0] * len(files)
        if is_bird_dir:
            num_birds += len(files)
            print(par_dir, is_bird_dir)
        else:
            num_not_birds += len(files)
    print(len(labels), num_not_birds, num_birds)
    with open('../BirdCheckerNet/labels.pk', 'wb') as file:
        pickle.dump(labels, file)
    
    print('Sifted all the data')


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