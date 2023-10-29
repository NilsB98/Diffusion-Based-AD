from typing import TypedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

import feature_extraction
from pipeline_reconstruction_ddim import DDIMReconstructionPipeline
from utils.anomalies import DiffMaps


def generate_samples(model, noise_scheduler, extractor, original_images, eta, steps_to_regenerate, start_at_timestep,
                     patch_imgs=False, noise_kind='gaussian', fl_smoothing_kernel_size=3):
    num_imgs = len(original_images)
    if patch_imgs:
        original_images = split_batch_into_patch(original_images, model.sample_size)

    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
        noise_kind=noise_kind
    )

    original_images = original_images.to(model.device)
    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    pipe_output = pipeline(
        batch_size=len(original_images),
        generator=generator,
        num_inference_steps=steps_to_regenerate,
        original_images=original_images,
        eta=eta,
        start_at_timestep=start_at_timestep,
        output_type="torch",
    )
    reconstruction = pipe_output.images
    history = pipe_output.history

    original = unnormalize_original_img(original_images)

    if patch_imgs:
        reconstruction = stitch_batch_patches(reconstruction, num_imgs)
        original = stitch_batch_patches(original, num_imgs)

    diff_maps = create_diffmaps(original, reconstruction, extractor, model.sample_size, fl_smoothing_kernel_size)
    history["images"] = [output_to_img(output, num_imgs) for output in history["images"]]

    return original.cpu(), reconstruction.cpu(), diff_maps, history


def create_diffmaps(original, reconstruction, extractor, extractor_resolution: int, fl_smoothing_size=3) -> DiffMaps:
    with torch.no_grad():
        diff_maps:DiffMaps = {}

        # pixel-level
        diff_map = (original - reconstruction) ** 2
        pl_diff_map = torch.amax(diff_map, (1))[:, None, :, :]
        diff_maps['diffmap_pl'] = pl_diff_map

        # feature-level
        num_imgs = len(original)
        original = split_batch_into_patch(original, extractor_resolution)
        reconstruction = split_batch_into_patch(reconstruction, extractor_resolution)

        if extractor is not None:
            resnet_diffmap = feature_extraction.utils.create_fl_diffmap(extractor, original, reconstruction, fl_smoothing_size)
            resnet_diffmap = stitch_batch_patches(resnet_diffmap, num_imgs)
            diff_maps['diffmap_fl'] = resnet_diffmap

        return diff_maps


def plot_single_channel_imgs(imgs, titles, cmaps=None, vmaxs=None, save_to=None, show_img=False):
    if cmaps is None:
        cmaps = ['viridis' for _ in imgs]

    fig, axs = plt.subplots(nrows=1, ncols=len(imgs))
    fig.set_figwidth(15)
    for i, (img, title, cmap, vmax) in enumerate(zip(imgs, titles, cmaps, vmaxs)):
        aximg = axs[i].imshow(img.squeeze(), cmap=cmap, vmin=0, vmax=vmax)
        axs[i].set_title(title)
        axs[i].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        fig.colorbar(aximg, ax=axs[i], shrink=1)

    if save_to is not None:
        plt.savefig(save_to)

    if show_img:
        plt.show()
    plt.close('all')


def plot_rgb_imgs(imgs, titles, save_to=None, show_img=False):
    fig, axs = plt.subplots(nrows=1, ncols=len(imgs))
    fig.set_figwidth(15)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axs[i].imshow(img.squeeze().permute(1, 2, 0))
        axs[i].set_title(title)
        axs[i].tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    if save_to is not None:
        plt.savefig(save_to)
    if show_img:
        plt.show()
    plt.close('all')


def show(imgs, title):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.title(title)
    plt.show()


def output_to_img(output, num_imgs):
    img = stitch_batch_patches(output, num_imgs)
    return img.cpu()


def unnormalize_original_img(original_image):
    original_image = transforms.Normalize([-0.5 * 2], [2])(original_image)
    return original_image


def gray_to_rgb(image: torch.Tensor):
    return image.repeat((1, 3, 1, 1))


def split_batch_into_patch(batch: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Takes a batch of images and splits each of them into a batch of images.
    The patches belonging to one image will be next to each other.

    :param batch: Batch of images to split with shape (B, CH, H, W)
    :param patch_size: Size of the patches to be created.
    :return: batch of patches shape: (B*(H/patch_size)**2), CH, patch_size, patch_size)
    """

    patches = [split_into_patches(img, patch_size) for img in torch.unbind(batch)]
    return torch.cat(patches)


def split_into_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Takes a single image and splits it into a batch of patches.
    I.e. a new dimension is added.

    :param image:  Image to split
    :param patch_size: Size which the image patches should have (rectangular)
    :return: batch of image patches => shape: (B, CH, patch_size, patch_size)
    """

    ps = patch_size
    assert len(image.shape) == 3, f"Image is expected to have shape (CH,H,W). Given: {image.shape}"
    assert image.shape[-1] % ps == 0, f"image of shape {image.shape} cannot be split into patches with patch size {ps}"

    # differentiate between rgb and grayscale images:
    n_ch = image.shape[0]
    patches = image.data.unfold(0, n_ch, n_ch).unfold(1, ps, ps).unfold(2, ps, ps).reshape(-1, n_ch, ps, ps)

    return patches


def stitch_batch_patches(patches: torch.Tensor, num_imgs: int) -> torch.Tensor:
    ppi = patches.shape[0] // num_imgs  # number of patches per image
    img_list = [stitch_patches(patches[ppi * i:ppi * (i + 1)]) for i in range(num_imgs)]
    return torch.cat(img_list)


def stitch_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Stitch patches of images back together. (Inverse operation to @split_into_patches)

    :param patches: either a 2D or 1D batch of image patches. Assume 1D for img_patches of 4 dims and 2D for 5 dims.
    :return: stitched image tensor with batch size of 1 (1, 3, H, W).
    """

    # convert to 2D if its in 1D batch
    if len(patches.shape) == 4:
        dim = int(patches.shape[0] ** 0.5)
        patches = torch.reshape(patches, (dim, dim, *patches.shape[1:]))

    rows = [torch.cat(patches[i].unbind(), dim=2) for i in range(len(patches))]

    # Now, concatenate the rows along the vertical axis (axis 2)
    stitched = torch.cat(rows, dim=1)

    return stitched.unsqueeze(0)


def add_batch_overlay(imgs: torch.Tensor, overlays: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for add_overlay() to add an overlay to each image

    :param imgs: batch of images
    :param overlays: batch of single-channel overlays
    :return: batch of tensors with blend of image and overlay
    """

    return torch.stack([add_overlay(img, overlay) for (img, overlay) in zip(imgs, overlays)])


def add_overlay(img: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
    """
    Add a heatmap-like overlay to an image to make anomalous regions more visible.

    :param img: original rgb-image
    :param overlay: single-channel overlay or anomaly map
    :return: tensor with blend of image and overlay
    """

    assert len(overlay.shape) == len(img.shape), "overlay and img must have the same number of dimensions"
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        overlay = overlay.unsqueeze(0)

    to_tensor = F.pil_to_tensor if img.max() > 1 else F.to_tensor

    heatmap = torch.zeros_like(overlay)
    heatmap = torch.cat((overlay, heatmap, heatmap), dim=1)

    img_list = torch.unbind(img)
    pil_imgs = [F.to_pil_image(img) for img in img_list]

    heatmap_list = torch.unbind(heatmap)
    pil_heatmaps = [F.to_pil_image(heatmap.to(torch.float)) for heatmap in heatmap_list]

    pil_blends = [Image.blend(pil_img, pil_hm, 0.5) for (pil_img, pil_hm) in zip(pil_imgs, pil_heatmaps)]
    blends = torch.cat([to_tensor(blend) for blend in pil_blends])
    return blends
