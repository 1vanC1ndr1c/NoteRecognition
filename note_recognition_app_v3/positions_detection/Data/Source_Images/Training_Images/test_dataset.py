import os
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path


def main():
    input_images_path = os.path.join(str(Path(__file__).parent), 'vott-csv-export')
    input_images = [f for f in listdir(input_images_path)
                    if isfile(join(input_images_path, f))
                    and f.endswith('.png')]

    csv = [f for f in listdir(input_images_path)
           if isfile(join(input_images_path, f))
           and f.endswith('.csv')][0]

    txt = [f for f in listdir(input_images_path)
           if isfile(join(input_images_path, f))
           and f.endswith('.txt')][0]

    with open(txt, mode="r", newline=None) as input_file:
        txt_input_data = [x.strip() for x in input_file.readlines()]

    txt_img_names = [x.split(' ')[0]
                     for x in [str(x.split('\\')[-1]).strip()
                               for x in txt_input_data
                               if '.png' in x]
                     if 'png' in x.split(' ')[0]]

    with open(csv, mode="r", newline=None) as input_file:
        csv_input_data = [x.strip() for x in input_file.readlines()]

    csv_images = set([x.split(',')[0].replace("'", "").replace('"', '')
                      for x in csv_input_data if 'png' in x])

    for img in input_images:
        if img not in csv_images:
            print(f'img not in csv file={img}')
        if img not in txt_img_names:
            print(f'img not in txt file={img}')

    print(f'LEN IMGS = {len(input_images)}, '
          f'csv = {csv}, csvLEN={len(csv_images)}, '
          f'txt={txt}, txtLEN={len(txt_input_data)}')


if __name__ == '__main__':
    sys.exit(main())
