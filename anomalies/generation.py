import pathlib
import pathlib
import random

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import ColorJitter, RandomRotation

from noise.simplex import simplexGenerator
from utils.visualize import split_into_patches, stitch_patches


class CorruptionGenerator:
    def __init__(self, path_to_anomaly_textures, num_patches=8):
        self.predefined_anomalies = self.load_predefined_img_anomalies(path_to_anomaly_textures)
        self.num_patches = num_patches
        self.color_trans = ColorJitter(.2, 0, .2, .5)
        self.rot_trans = RandomRotation(180)


    def generate_corruption(self, img: Tensor, is_heterologous_anomaly=True):
        """
        Heterologous anomalies are such where the original image is just transformed

        :param img:
        :param is_heterologous_anomaly:
        :return:
        """
        if is_heterologous_anomaly:     # TODO extend for batches
            patch_size = img.shape[-1] // self.num_patches
            img = self.rot_trans(img)
            patches = split_into_patches(img, patch_size)
            patches = patches[torch.randperm(len(patches))]
            patches = self.color_trans(patches)
            corruption_img = stitch_patches(patches)
            return corruption_img[0]
        else:
            idx = random.randint(0, len(self.predefined_anomalies))
            anomaly = self.predefined_anomalies[idx]
            smaller_side: int = min(list(anomaly.shape[-2:]))
            anomaly = F.center_crop(anomaly, [smaller_side, smaller_side])
            anomaly = F.resize(anomaly, [img.shape[-1], img.shape[-1]], antialias=True)
            return anomaly / 255.

    def __call__(self, img, is_heterologous_anomaly=True, *args, **kwargs):
        return self.generate_corruption(img, is_heterologous_anomaly)

    @staticmethod
    def load_predefined_img_anomalies(path, extension='jpg'):
        image_paths = pathlib.Path(path).glob("*." + extension)
        images = [read_image(str(p)) for p in image_paths]

        return images


class ImageCorruptor:
    def __init__(self, corruption_generator: CorruptionGenerator, is_texture=True):
        self.is_texture = is_texture    # TODO for later versions where the anomalies are only placed on the object
        self.corruption_generator = corruption_generator

    def corrupt_img(self, img, transparency):
        b = transparency
        noise = simplexGenerator.batch_2d_octaves(img.shape, 6, 0.6)

        allowed_regions = torch.ones_like(img)  # defines where anomalies are allowed
        proposed_regions = torch.where(noise > 0.4, 1, 0) # defines at which places the anomalies should be placed
        anomaly_regions = allowed_regions * proposed_regions
        no_anomaly_regions = torch.abs(anomaly_regions - 1)

        corruptions = self.corruption_generator(img, False)

        corrupted_img = no_anomaly_regions * img + (1-b) * (anomaly_regions * corruptions) + b * (anomaly_regions * img)

        return corrupted_img

    def __call__(self, img, transparency):
        return self.corrupt_img(img, transparency)



