# imports
import json
import os.path
import time
import argparse

from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from pipe.train import train_step
from pipe.validate import validate_step
from diffusers import DDPMScheduler, UNet2DModel, get_scheduler
from tqdm import tqdm
from loader.loader import MVTecDataset
from utils.files import save_args
from utils.visualize import generate_samples
from dataclasses import dataclass
from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DBADScheduler


@dataclass
class TrainArgs:
    checkpoint_dir: str
    log_dir: str
    run_name: str
    mvtec_item: str
    flip: bool
    rotate: bool
    color_jitter: bool
    resolution: int
    epochs: int
    save_n_epochs: int
    dataset_path: str
    train_steps: int
    beta_schedule: str
    device: str
    reconstruction_weight: float
    eta: float
    batch_size: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='directory path to store the checkpoints')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--run_name', type=str, required=True,
                        help='name of the run and corresponding checkpoints/logs that are created')
    parser.add_argument('--mvtec_item', type=str, required=True,
                        choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                                 "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--resolution', type=int, default=128,
                        help='resolution of the images to generate (dataset will be resized to this resolution during training)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs to train for')
    parser.add_argument('--flip', action='store_true',
                        help='whether to augment training data with a flip')
    parser.add_argument('--rotate', type=float, default=0,
                        help='degree of rotation to augment training data with')
    parser.add_argument('--color_jitter', type=float, default=0,
                        help='amount of color jitter to augment training data with')
    parser.add_argument('--save_n_epochs', type=int, default=50,
                        help='write a checkpoint every n-th epoch')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='number of steps for the full diffusion process')
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the (mvtec) dataset')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during inference (doesnt affect training)')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness. Only used during inference, not training.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size during training')

    return TrainArgs(**vars(parser.parse_args()))


def transform_imgs_test(imgs):
    augmentations = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.CenterCrop(args.resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def transform_imgs_train(imgs):
    augmentations = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if args.flip else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(args.rotate),
            transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
            transforms.ToTensor(),
            transforms.CenterCrop(args.resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def main(args: TrainArgs):
    # -------------      load data      ------------
    data_train = MVTecDataset(args.dataset_path, True, args.mvtec_item, ["good"],
                              transform_imgs_train)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, ["all"],
                             transform_imgs_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # ----------- set model, optimizer, scheduler -----------------
    model_args = {
        "sample_size": args.resolution,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "block_out_channels": (128, 128, 256, 256, 512, 512),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
    }
    model = UNet2DModel(
        **model_args
    )

    noise_scheduler = DDPMScheduler(args.train_steps, beta_schedule=args.beta_schedule)
    # noise_scheduler = DBADScheduler(args.train_steps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-6,
        lr=1e-4,
        betas=(0.95, 0.999),
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * args.epochs),
    )
    loss_fn = torch.nn.MSELoss()

    # additional info/util
    timestamp = str(time.time())[:11]
    writer = SummaryWriter(f'{args.log_dir}/{args.run_name}_{timestamp}')

    # -----------------     train loop   -----------------
    print("**** starting training *****")
    print(f"run_id: {args.run_name}_{timestamp}")
    save_args(args, f"{args.checkpoint_dir}/{args.run_name}_{timestamp}", "train_arg_config")
    save_args(model_args, f"{args.checkpoint_dir}/{args.run_name}_{timestamp}", "model_config")

    for epoch in range(args.epochs):
        model.train()
        model.to(args.device)

        progress_bar = tqdm(total=len(train_loader) + len(test_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss_train = 0

        for btc_num, (batch, label) in enumerate(train_loader):
            loss = train_step(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, args.train_steps, 'perlin')

            running_loss_train += loss
            progress_bar.update(1)

        running_loss_test = 0
        with torch.no_grad():
            for btc_num, (batch, label, gt) in enumerate(test_loader):
                loss = validate_step(model, batch, noise_scheduler, args.train_steps, loss_fn)

                running_loss_test += loss
                progress_bar.update(1)

            progress_bar.set_postfix_str(
                f"Train Loss: {running_loss_train / len(train_loader)}, Test Loss: {running_loss_test / len(test_loader)}")
            progress_bar.close()

            if epoch % 100 == 0:
                # generate images
                # TODO use method from inference script
                noise_scheduler_inference = DDIMScheduler(args.train_steps, beta_schedule=args.beta_schedule,
                                                          reconstruction_weight=args.reconstruction_weight)
                train_grid, train_mask = generate_samples(model, noise_scheduler_inference, f"Train samples {epoch=}",
                                                          next(iter(train_loader))[0], args.eta, steps_to_regenerate=20,
                                                          start_at_timestep=200)
                test_grid, test_mask = generate_samples(model, noise_scheduler_inference, f"Test samples {epoch=}",
                                                        next(iter(test_loader))[0], args.eta, steps_to_regenerate=20,
                                                        start_at_timestep=200)

                writer.add_image(f'Test samples', test_grid, epoch)

            if epoch % args.save_n_epochs == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}_{timestamp}/epoch_{epoch}.pt")

        writer.add_scalar('Loss/train', running_loss_train, epoch)
        writer.add_scalar('Loss/test', running_loss_test, epoch)

    writer.add_hparams({'category': args.mvtec_item, 'res': args.resolution, 'eta': args.eta,
                        'recon_weight': args.reconstruction_weight}, {'MSE': running_loss_test},
                       run_name='hp')

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}_{timestamp}/epoch_{args.epochs}.pt")


if __name__ == '__main__':
    args: TrainArgs = parse_args()
    main(args)
