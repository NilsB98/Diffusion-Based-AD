# imports
import argparse
import os.path
import time
from collections import Counter
from dataclasses import dataclass

import diffusers
import torch
from diffusers import DDPMScheduler, UNet2DModel, get_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import feature_extraction
import inference_ddim
import pipe.inference
from loader.loader import MVTecDataset
from pipe.train import train_step
from pipe.validate import validate_step
from schedulers.scheduling_ddim import DDIMScheduler
from utils.files import save_args


@dataclass
class TrainArgs:
    checkpoint_dir: str
    run_name: str
    mvtec_item: str
    flip: bool
    rotate: float
    color_jitter: float
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
    noise_kind: str
    crop: bool
    log_dir: str
    img_dir: str
    plt_imgs: bool
    calc_val_loss: bool
    extractor_path: str
    checkpoint_name: str


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='directory path to store the checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default="final.pt",
                        help='Name of the final checkpoint to be stored.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the run and corresponding log and checkpoint directory that will be created.')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store the checkpoints')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='directory path to store the generated images in. Will create a new sub-directory.')
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
    parser.add_argument('--extractor_path', type=str,
                        help='Path to the feature extractor. This is extractor is used to calculate differences between original and reconstructed images in a deep learning fashion.')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during inference (doesnt affect training)')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness. Only used during inference, not training.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size during training')
    parser.add_argument('--noise_kind', type=str, default="gaussian",
                        choices=["simplex", "gaussian"],
                        help='Kind of noise to use for the noising steps.')
    parser.add_argument('--crop', action='store_true',
                        help='If set: the image will be cropped to the resolution instead of resized.')
    parser.add_argument('--plt_imgs', action='store_true',
                        help='If set: Plot images via matplotlib')
    parser.add_argument('--calc_val_loss', action='store_true',
                        help='If set: Calculate the validation loss as well as the train loss.')

    return TrainArgs(**vars(parser.parse_args()))


def transform_imgs_test(imgs, args):
    augmentations = transforms.Compose(
        [
            transforms.RandomCrop(args.resolution) if args.crop else transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def transform_imgs_train(imgs, args):
    augmentations = transforms.Compose(
        [
            transforms.RandomCrop(args.resolution) if args.crop else transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if args.flip else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(args.rotate),
            transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]


def main(args: TrainArgs):
    # -------------      load data      ------------
    data_train = MVTecDataset(args.dataset_path, True, args.mvtec_item, ["good"],
                              lambda x: transform_imgs_train(x, args))
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, ["all"],
                             lambda x: transform_imgs_test(x, args))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # ----------- set model, optimizer, scheduler -----------------
    channel_multiplier = {
        128: (128, 128, 256, 384, 512),
        256: (128, 128, 256, 256, 512, 512)
    }
    down_blocks = ["DownBlock2D" for _ in channel_multiplier[args.resolution]]
    down_blocks[-2] = "AttnDownBlock2D"
    up_blocks = ["UpBlock2D" for _ in channel_multiplier[args.resolution]]
    up_blocks[1] = "AttnUpBlock2D"

    model_args = {
        "sample_size": args.resolution,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "block_out_channels": channel_multiplier[args.resolution],
        "down_block_types": down_blocks,
        "up_block_types": up_blocks
    }
    model = UNet2DModel(
        **model_args
    )

    noise_scheduler = DDPMScheduler(args.train_steps, beta_schedule=args.beta_schedule)
    inf_noise_scheduler = DDIMScheduler(args.train_steps, 150,
                                        beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                        reconstruction_weight=args.reconstruction_weight, noise_type=args.noise_kind)

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

    # extractor = feature_extraction.ResNetFE(args.extractor_path)
    # extractor.eval()
    # extractor.to(args.device)

    # additional info/util
    writer = SummaryWriter(f'{args.log_dir}/{args.run_name}')
    diffmap_blur = transforms.GaussianBlur(2 * int(4 * 4 + 0.5) + 1, 4)
    print(diffusers.utils.logging.is_progress_bar_enabled())
    diffusers.utils.logging.disable_progress_bar()

    # -----------------     train loop   -----------------
    print("**** starting training *****")
    print(f"run name: {args.run_name}")
    save_args(args, f"{args.checkpoint_dir}/{args.run_name}", "train_arg_config")
    save_args(model_args, f"{args.checkpoint_dir}/{args.run_name}", "model_config")

    for epoch in range(args.epochs):
        model.train()
        model.to(args.device)

        progress_bar_len = len(train_loader) + len(test_loader) if args.calc_val_loss else len(train_loader)
        progress_bar = tqdm(total=progress_bar_len)
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss_train = 0

        for btc_num, (batch, _) in enumerate(train_loader):
            loss = train_step(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, args.train_steps, args.noise_kind)

            running_loss_train += loss
            progress_bar.update(1)

        running_loss_test = 0
        with torch.no_grad():
            scores = Counter()
            for _btc_num, (_batch, _labels, gts) in enumerate(test_loader):
                loss = validate_step(model, _batch, noise_scheduler, args.train_steps, loss_fn, args.noise_kind) if args.calc_val_loss else 0

                running_loss_test += loss

                writer.add_scalars(main_tag='scores', tag_scalar_dict=dict(scores), global_step=epoch)
                progress_bar.update(1)

                if epoch % 100 == 0:
                    pipe.inference.run_inference_step(None, diffmap_blur, scores, gts, f"ep{epoch}_btc{_btc_num}", _batch, model,
                                                      args.noise_kind, inf_noise_scheduler, _labels, writer, args.eta, 25,
                                                      250, args.crop, args.plt_imgs, os.path.join(args.img_dir, args.run_name))

            for key in scores:
                scores[key] /= len(test_loader)

            progress_bar.set_postfix_str(
                f"Train Loss: {running_loss_train / len(train_loader)}, Test Loss: {running_loss_test / len(test_loader)}, {dict(scores)}")
            progress_bar.close()

            if epoch % args.save_n_epochs == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}/epoch_{epoch}.pt")

        writer.add_scalar('Loss/train', running_loss_train, epoch)
        writer.add_scalar('Loss/test', running_loss_test, epoch)

    writer.add_hparams({'category': args.mvtec_item, 'res': args.resolution, 'eta': args.eta,
                        'recon_weight': args.reconstruction_weight}, {'MSE': running_loss_test},
                       run_name='hp')

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), f"{args.checkpoint_dir}/{args.run_name}/{args.checkpoint_name}")


if __name__ == '__main__':
    args: TrainArgs = parse_args()
    main(args)
