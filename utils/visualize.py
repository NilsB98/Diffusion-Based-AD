import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from PIL import Image

from pipeline_reconstruction_ddim import DDIMReconstructionPipeline


def generate_samples(model, noise_scheduler, plt_title, original_images, eta, steps_to_regenerate, start_at_timestep, show_plt=True):
    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    images = pipeline(
        batch_size=8,
        generator=generator,
        num_inference_steps=steps_to_regenerate,
        original_images=original_images.to(model.device),
        eta=eta,
        start_at_timestep=start_at_timestep,
        output_type="numpy",
    ).images

    images_processed = (images * 255).round().astype("int")
    images = torch.from_numpy(images_processed)
    # images = torch.permute(images, (0, 3, 1, 2))

    original_images = transforms.Normalize([-0.5 * 2], [2])(original_images)
    originals = (original_images * 255).round().type(torch.int32)

    diff_map = (originals - images) ** 2
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)  # per channel and image
    diff_map = (diff_map * 255).round()
    diff_map = transforms.functional.rgb_to_grayscale(diff_map).to(torch.uint8)

    grid_generated_imgs = make_grid(torch.cat((images.to(torch.uint8), originals.to(torch.uint8)), 0), 4)
    grid_mask = make_grid(diff_map, 4)
    plt.imshow(transforms.functional.rgb_to_grayscale(grid_mask).to(torch.uint8).squeeze(), cmap='viridis')
    plt.colorbar()
    plt.show()
    if show_plt:
        show(grid_generated_imgs, plt_title)
        show(grid_mask, plt_title + ' mask')

    return grid_generated_imgs, grid_mask


def generate_single_sample(model, noise_scheduler, original_image, eta, steps_to_regenerate, start_at_timestep, patch_imgs=False, noise_kind='gaussian'):
    if patch_imgs:
        original_image = split_into_patches(original_image[0], model.sample_size)

    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
        noise_kind=noise_kind
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    pipe_output = pipeline(
        batch_size=1,
        generator=generator,
        num_inference_steps=steps_to_regenerate,
        original_images=original_image.to(model.device),
        eta=eta,
        start_at_timestep=start_at_timestep,
        output_type="numpy",
    )
    reconstruction = pipe_output.images
    history = pipe_output.history

    # images_processed = (reconstruction * 255).round().astype("int")
    reconstruction = torch.from_numpy(reconstruction)

    original = unnormalize_original_img(original_image)

    if patch_imgs:
        reconstruction = stitch_patches(reconstruction)
        original = stitch_patches(original)

    # TODO try using (diff_map > 0.5).float() directly as the anomaly map => i.e. no diffmap to plot
    diff_map = (original - reconstruction) ** 2  # TODO try diff in HSV instead of RGB
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)  # per channel and image
    # diff_map = transforms.functional.rgb_to_grayscale(diff_map)  # TODO to 1D by taking max of the 3 channels,
    diff_map = torch.amax(diff_map, (1))[:, None, :, :]
    history["images"] = [output_to_img(output) for output in history["images"]]

    return original, reconstruction, diff_map, history


def plot_single_channel_imgs(imgs, titles, cmaps=None, save_to=None, show_img=False):
    if cmaps is None:
        cmaps = ['viridis' for _ in imgs]

    fig, axs = plt.subplots(nrows=1, ncols=len(imgs))
    fig.set_figwidth(15)
    for i, (img, title, cmap) in enumerate(zip(imgs, titles, cmaps)):
        aximg = axs[i].imshow(img.squeeze(), cmap=cmap)
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


def output_to_img(output):
    # img = (output * 255).round().astype("int")
    img = torch.from_numpy(output)
    img = stitch_patches(img)
    return img


def unnormalize_original_img(original_image):
    original_image = transforms.Normalize([-0.5 * 2], [2])(original_image)
    # original = (original_image * 255).round().type(torch.int32)
    return original_image


def gray_to_rgb(image: torch.Tensor):
    return image.repeat((1, 3, 1, 1))

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
