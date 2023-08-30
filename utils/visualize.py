import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

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
    images = torch.permute(images, (0, 3, 1, 2))

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


def generate_single_sample(model, noise_scheduler, original_image, eta, steps_to_regenerate, start_at_timestep):
    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
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

    images_processed = (reconstruction * 255).round().astype("int")
    reconstruction = torch.from_numpy(images_processed)
    reconstruction = torch.permute(reconstruction, (0, 3, 1, 2))

    original_image = transforms.Normalize([-0.5 * 2], [2])(original_image)
    original = (original_image * 255).round().type(torch.int32)

    diff_map = (original - reconstruction) ** 2
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)  # per channel and image
    diff_map = (diff_map * 255).round()
    diff_map = transforms.functional.rgb_to_grayscale(diff_map).to(torch.uint8)

    history = [output_to_img(output) for output in history]

    return original, reconstruction, diff_map, history


def plot_single_channel_imgs(imgs, titles, save_to=None, show_img=False):
    fig, axs = plt.subplots(nrows=1, ncols=len(imgs))
    fig.set_figwidth(15)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        aximg = axs[i].imshow(img.squeeze(), cmap='viridis')
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
    img = (output * 255).round().astype("int")
    img = torch.from_numpy(img)
    img = torch.permute(img, (0, 3, 1, 2)).to(torch.uint8)
    return img


def gray_to_rgb(image: torch.Tensor):
    return image.repeat((1, 3, 1, 1))
