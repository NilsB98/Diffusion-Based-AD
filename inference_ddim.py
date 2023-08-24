# imports
import argparse
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from loader.loader import MVTecDataset
from pipeline_reconstruction_ddim import DDIMReconstructionPipeline
from schedulers.scheduling_ddim import DDIMScheduler

@dataclass
class InferenceArgs:
    steps_to_regenerate: int
    reconstruction_weight: float
    mvtec_item: str
    mvtec_item_states: list
    checkpoint_dir: str
    checkpoint_name: str
    log_dir: str
    train_steps: int
    beta_schedule: str
    flip: bool
    eta: float
    device: str
    dataset_path: str
    shuffle: bool

def parse_args() -> InferenceArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='directory path to store the checkpoints')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='name of the run and corresponding checkpoints/logs that are created')
    parser.add_argument('--mvtec_item', type=str, required=True, choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--mvtec_item_states', type=str, nargs="+", required=True,
                        help="States of the mvtec items that should be used. Available options depend on the selected item.")
    parser.add_argument('--flip', action='store_true',
                        help='whether to augment training data with a flip')
    parser.add_argument('--steps_to_regenerate', type=int, default=300,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='number of steps for the full diffusion process')
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the (mvtec) dataset')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during generation')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the items in the dataset')

    return InferenceArgs(**vars(parser.parse_args()))

def generate_samples(model, noise_scheduler, plt_title, original_images, eta, steps_to_regenerate):
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

    grid = make_grid(torch.cat((images.to(torch.uint8), originals.to(torch.uint8)), 0), 4)
    show(grid, plt_title)
    grid = make_grid(diff_map, 4)
    show(grid, plt_title + ' mask')


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


def main(args: InferenceArgs):
    # train loop
    print("**** starting inference *****")
    config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
    model_config = json.loads(config_file.read())

    augmentations = transforms.Compose(
        [
            transforms.Resize(model_config["sample_size"], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if args.flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(imgs):
        return [augmentations(image.convert("RGB")) for image in imgs]

    # data loader
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, args.mvtec_item_states,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=args.shuffle)

    # set model, optimizer, scheduler
    model = UNet2DModel(
        **model_config
    )

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        # validate and generate images
        noise_scheduler_inference = DDIMScheduler(args.train_steps, beta_schedule=args.beta_schedule, reconstruction_weight=args.reconstruction_weight)
        generate_samples(model, noise_scheduler_inference, f"Test samples ", next(iter(test_loader))[0], args.eta, args.steps_to_regenerate)


if __name__ == '__main__':
    args: InferenceArgs = parse_args()
    main(args)
