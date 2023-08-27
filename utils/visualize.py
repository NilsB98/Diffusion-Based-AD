import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from pipeline_reconstruction_ddim import DDIMReconstructionPipeline


def generate_samples(model, noise_scheduler, plt_title, original_images, eta, steps_to_regenerate, show_plt=True):
    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        batch_size=8,
        generator=generator,
        num_inference_steps=100,    # depending on this the number of skipped timesteps is calculated
        original_images=original_images.to(model.device),
        eta=eta,
        start_at_timestep=steps_to_regenerate,
        output_type="numpy",
    ).images

    images_processed = (images * 255).round().astype("int")
    images = torch.from_numpy(images_processed)
    images = torch.permute(images, (0, 3, 1, 2))

    original_images = transforms.Normalize([-0.5 * 2], [2])(original_images)
    originals = (original_images * 255).round().type(torch.int32)

    diff_map = (originals - images) ** 2
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)   # per channel and image
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

def generate_single_sample(model, noise_scheduler, plt_title, original_image, eta, steps_to_regenerate, img_dir=None, show_plt=True):
    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    image = pipeline(
        batch_size=1,
        generator=generator,
        num_inference_steps=100,    # depending on this the number of skipped timesteps is calculated
        original_images=original_image.to(model.device),
        eta=eta,
        start_at_timestep=steps_to_regenerate,
        output_type="numpy",
    ).images

    images_processed = (image * 255).round().astype("int")
    image = torch.from_numpy(images_processed)
    image = torch.permute(image, (0, 3, 1, 2))

    original_image = transforms.Normalize([-0.5 * 2], [2])(original_image)
    original = (original_image * 255).round().type(torch.int32)

    diff_map = (original - image) ** 2
    diff_map = diff_map / torch.amax(diff_map, dim=(2, 3)).reshape(-1, 3, 1, 1)   # per channel and image
    diff_map = (diff_map * 255).round()
    diff_map = transforms.functional.rgb_to_grayscale(diff_map).to(torch.uint8)

    plt.imshow(diff_map.squeeze(), cmap='viridis')
    plt.colorbar()
    plt.title(plt_title + ' - heatmap')
    if show_plt:
        plt.show()
    else:
        plt.savefig(f"{img_dir}/{plt_title}_heatmap.png")
        plt.close('all')

    plt.imshow(torch.concat((original.squeeze().permute(1, 2, 0), image.squeeze().permute(1, 2, 0)), 1))
    plt.title(plt_title)
    if show_plt:
        plt.show()
    else:
        plt.savefig(f"{img_dir}/{plt_title}.png")
        plt.close('all')

    return image, diff_map


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
