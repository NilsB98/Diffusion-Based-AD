import torch
from torch import Tensor
from utils.visualize import split_into_patches, stitch_patches
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F

from noise.simplex import simplexGenerator

class CorruptionGenerator:
    def __init__(self, path_to_anomaly_textures, num_patches=8):
        # TODO load anomalies in RAM
        self.num_patches = num_patches
        self.color_trans = ColorJitter(.2, 0, .2, .5)

    def generate_corruption(self, img: Tensor, is_heterologous_anomaly=True):
        """
        Heterologous anomalies are such where the original image is just transformed

        :param img:
        :param is_heterologous_anomaly:
        :return:
        """
        if is_heterologous_anomaly:
            patch_size = img.shape[-1] // self.num_patches
            patches = split_into_patches(img, patch_size)
            patches = patches[torch.randperm(len(patches))]
            corruption_img = stitch_patches(patches)
            corruption_img = F.adjust_hue(corruption_img[0], .2)  # self.color_trans(patches)
            return corruption_img
        else:
            raise NotImplementedError()

    def __call__(self, img, is_heterologous_anomaly=True, *args, **kwargs):
        return self.generate_corruption(img, is_heterologous_anomaly)


class ImageCorruptor:
    def __init__(self, corruption_generator: CorruptionGenerator, is_texture=True):
        self.is_texture = is_texture    # TODO for later versions where the anomalies are only placed on the object
        self.corruption_generator = corruption_generator

    def corrupt_img(self, img, transparency):
        noise = simplexGenerator.rand_2d_octaves(img.shape, 6, 0.6)

        allowed_regions = torch.ones_like(img)  # defines where anomalies are allowed
        proposed_regions = torch.where(noise > 0, 1, 0)# defines at which places the anomalies should be placed
        anomaly_regions = allowed_regions * proposed_regions

        return



