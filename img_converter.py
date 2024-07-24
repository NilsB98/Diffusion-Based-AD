import os
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import Pool
import datetime

def convert_single_image(img_data):
    image_dir, output_dir, img_name = img_data
    image = Image.open(os.path.join(image_dir, img_name))
    image.save(os.path.join(output_dir, img_name[:-4] + '.png'), 'PNG')
    print("Saved image", img_name[:-4] + '.png')


def convert_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)


    images_to_convert = [(image_dir, output_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.bmp') and not is_target_existent(output_dir, img_name)]

    print(len(images_to_convert))

    with Pool() as pool:
        pool.map(convert_single_image, images_to_convert)

def filter_files(start_time, end_time, directory):
    start_time = datetime.datetime.strptime(start_time, "%Y_%m_%d__%H_%M_%S")
    end_time = datetime.datetime.strptime(end_time, "%Y_%m_%d__%H_%M_%S")
    filtered_files = []

    for filename in os.listdir(directory):
        try:
            date_str = filename.split('_NOK')[0]
            file_time = datetime.datetime.strptime(date_str, "%Y_%m_%d__%H_%M_%S")
            if start_time <= file_time <= end_time:
                filtered_files.append(filename)
        except ValueError:
            # The filename didn't match the expected format, ignore it
            pass

    return filtered_files


def rename_files(files, old_core_num, new_core_num, directory):
    for filename in files:
        new_filename = filename.replace(old_core_num, new_core_num)
        print(f"{filename} -> {new_filename}")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


def rename_files_between(start_time, end_time, directory, old_core_num, new_core_num):
    files = filter_files(start_time, end_time, directory)
    print(f"files in timespan: {len(files)}")
    rename_files(files, old_core_num, new_core_num, directory)

def move_images(file_paths, destination):
    for file_path in file_paths:
        directory, filename = os.path.split(file_path)
        shutil.move(file_path, str(os.path.join(destination, filename)))

def is_target_existent(target_dir, target_name):
    return os.path.isfile(os.path.join(target_dir, target_name[:-3] + 'png'))

if __name__ == '__main__':
    # rename_files_between("2024_02_08__14_57_21", "2024_02_08__15_06_11", r"E:\Cores\NewCores\02\08", "0599M", "0594M")
    convert_images(r"F:\dc\data\03\12", r"E:\Cores\NewCores\03\12")
    convert_images(r"F:\dc\data\03\22", r"E:\Cores\NewCores\03\22")
    convert_images(r"F:\dc\data\03\25", r"E:\Cores\NewCores\03\25")
    paths_anomalies = [r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_16__14_31_13_NOK__S001_0385R_0301_Cam1.png',
r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_16__14_44_00_NOK__S001_0385R_10k_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_16__15_09_45_NOK__S001_0377R_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_18__12_16_52_NOK__S001_0377M_0301_Cam1.png',
                       r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_22__10_10_04_NOK__S001_0428M_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_01_22__10_32_46_NOK__S001_0462M_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_02_01__11_47_42_NOK__S001_0521R_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_02_01__14_09_20_NOK__S001_0521M_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_02_08__14_44_52_NOK__S001_0599M_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_02_08__15_28_30_NOK__S001_0614M_0301_Cam1.png',
           r'C:\Users\nilsb\Desktop\fullscale_data\core\train\good\2024_02_08__15_59_19_NOK__S001_0633M_0301_Cam1.png',
     ]
    # move_images(paths_anomalies, r"C:\Users\nilsb\Desktop\fullscale_data\core\test\crack")


