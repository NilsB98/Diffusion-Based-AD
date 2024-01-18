import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import Pool


def convert_single_image(img_data):
    image_dir, output_dir, img_name = img_data
    image = Image.open(os.path.join(image_dir, img_name))
    image.save(os.path.join(output_dir, img_name[:-4] + '.png'), 'PNG')
    print("Saved image", img_name[:-4] + '.png')


def convert_image(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)


    images_to_convert = [(image_dir, output_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.bmp') and not is_target_existent(output_dir, img_name)]

    print(len(images_to_convert))

    with Pool() as pool:
        pool.map(convert_single_image, images_to_convert)


def is_target_existent(target_dir, target_name):
    return os.path.isfile(os.path.join(target_dir, target_name[:-3] + 'png'))

if __name__ == '__main__':
    convert_image(r"F:\dc\data\01\16", r"E:\Cores\NewCores\01\16")
