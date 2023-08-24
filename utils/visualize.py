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
    if show_plt:
        show(grid_generated_imgs, plt_title)
        show(grid_mask, plt_title + ' mask')

    return grid_generated_imgs, grid_mask



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
