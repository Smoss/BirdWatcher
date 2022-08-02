import argparse

import imagenetLabels

DEFAULT_TARGET = './bird_names.py'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(
        '-o'
        '--output_file',
        help='File to output to',
        default=DEFAULT_TARGET
    )
    args = parser.parse_args()
    bird_names = []
    for name, _ in imagenetLabels.imagenet_labels.values():
        val = input(f'''Is {name} a type of bird?(y/N)''')
        if val.lower() == 'y':
            bird_names.append(name)
    with open(args.output_file, 'w') as file:
        file.writelines(['bird_names = ['] + bird_names + [']'])