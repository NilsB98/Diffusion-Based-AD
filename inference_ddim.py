# imports
import argparse
import json
from dataclasses import dataclass

import torch
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from loader.loader import MVTecDataset
from schedulers.scheduling_ddim import DDIMScheduler
from utils.files import save_args
from utils.visualize import generate_single_sample, plot_single_channel_imgs, plot_rgb_imgs, gray_to_rgb, \
    split_into_patches


@dataclass
class InferenceArgs:
    num_inference_steps: int
    start_at_timestep: int
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
    img_dir: str
    plt_imgs: bool
    patch_imgs: bool

def parse_args() -> InferenceArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='directory path to store the checkpoints')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--img_dir', type=str, default="generated_imgs",
                        help='directory path to store generated imgs')
    parser.add_argument('--checkpoint_name', type=str, required=True,
                        help='name of the run and corresponding checkpoints/logs that are created')
    parser.add_argument('--mvtec_item', type=str, required=True,
                        choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                                 "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--mvtec_item_states', type=str, nargs="+", default="all",
                        help="States of the mvtec items that should be used. Available options depend on the selected item. Set to 'all' to include all states")
    parser.add_argument('--flip', action='store_true',
                        help='whether to augment training data with a flip')
    parser.add_argument('--num_inference_steps', type=int, default=30,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--start_at_timestep', type=int, default=300,
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
    parser.add_argument('--plt_imgs', action='store_true',
                        help='Plot the images with matplot lib. I.e. call plt.show()')
    parser.add_argument('--patch_imgs', action='store_true',
                        help='If the image size is larger than the models input, split input into multiple patches and stitch it together afterwards.')

    return InferenceArgs(**vars(parser.parse_args()))


def main(args: InferenceArgs, writer: SummaryWriter):
    # train loop
    print("**** starting inference *****")
    config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
    model_config = json.loads(config_file.read())
    save_args(args, args.img_dir, "inference_args")

    augmentations = transforms.Compose(
        [
            # transforms.Resize(model_config["sample_size"], interpolation=transforms.InterpolationMode.BILINEAR) if not args.patch_imgs else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if args.flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            # transforms.CenterCrop(model_config["sample_size"]),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(imgs):
        return [augmentations(image.convert("RGB")) for image in imgs]

    # data loader
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, args.mvtec_item_states,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=args.shuffle)

    # set model, optimizer, scheduler
    model = UNet2DModel(
        **model_config
    )

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        # validate and generate images
        noise_scheduler_inference = DDIMScheduler(args.train_steps, args.start_at_timestep, beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                                  reconstruction_weight=args.reconstruction_weight)
        for i, (img, state, gt) in enumerate(test_loader):
            patches = split_into_patches(img[0], model_config["sample_size"])
            original, reconstruction, diffmap, history = generate_single_sample(model, noise_scheduler_inference, patches,
                                                                                args.eta, args.num_inference_steps,
                                                                                args.start_at_timestep)
            plot_single_channel_imgs([gt, diffmap], ["ground truth", "heatmap"],
                                     save_to=f"{args.img_dir}/{i}_{state[0]}_heatmap.png", show_img=args.plt_imgs)
            plot_rgb_imgs([original, reconstruction], ["original", "reconstructed"],
                          save_to=f"{args.img_dir}/{i}_{state[0]}.png", show_img=args.plt_imgs)

            writer.add_images(f"{i}_{state[0]}", torch.concat([original.to(torch.uint8)] + history + [gray_to_rgb(diffmap), gray_to_rgb(gt * 255).to(torch.uint8)]))


if __name__ == '__main__':
    args: InferenceArgs = parse_args()
    writer = SummaryWriter(f'{args.log_dir}/inference')
    main(args, writer)
