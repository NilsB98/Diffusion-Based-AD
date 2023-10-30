import argparse
import json
from dataclasses import dataclass
import os

import torch
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torchvision import transforms

from feature_extraction.autoencoder import Autoencoder, AETrainer, DBTrainer
from feature_extraction.extractor import ResNetFE
from loader.loader import MVTecDataset
from schedulers.scheduling_ddim import DDIMScheduler
from pipe.inference import generate_samples


@dataclass
class TrainArgs:
    checkpoint_dir: str
    item: str
    flip: bool
    resolution: int
    epochs: int
    dataset_path: str
    train_steps: int
    beta_schedule: str
    device: str
    reconstruction_weight: float
    eta: float
    batch_size: int
    noise_kind: str
    crop: bool
    checkpoint_name: str
    save_to: str
    start_at_timestep: int
    steps_to_regenerate: int
    use_diffusion_model: bool


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description='Add config for the training')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='directory path to store the checkpoints')
    parser.add_argument('--item', type=str, required=True,
                        choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                                 "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
                        help='name of the item within the MVTec Dataset to train on')
    parser.add_argument('--resolution', type=int, default=128,
                        help='resolution of the images to generate (dataset will be resized to this resolution during training)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='epochs to train for')
    parser.add_argument('--start_at_timestep', type=int, default=250,
                        help='Timestep from which the diffusion process should be started')
    parser.add_argument('--steps_to_regenerate', type=int, default=25,
                        help='Number of timesteps to generate during the DDIM process')
    parser.add_argument('--flip', action='store_true',
                        help='whether to augment training data with a flip')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='number of steps for the full diffusion process')
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the (mvtec) dataset')
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Checkpoint to load diffusion model from.')
    parser.add_argument('--save_to', type=str, required=True,
                        help='Full path and name to where the trained extractor should be saved (including .pt ending)')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness. Only used during inference, not training.')
    parser.add_argument('--reconstruction_weight', type=float, default=.1,
                        help='Influence of the original sample during diffusion')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size during training')
    parser.add_argument('--noise_kind', type=str, default="gaussian",
                        choices=["simplex", "gaussian"],
                        help='Kind of noise to use for the noising steps.')
    parser.add_argument('--crop', action='store_true',
                        help='If set: the image will be cropped to the resolution instead of resized.')
    parser.add_argument('--use_diffusion_model', action='store_true',
                        help='If not set the feature extractor will be trained in an AE-Fashion, s.t. the input image should match the output image. If set and a diffusion model is given the model is trained to reduce the distance between an input image and the diffusion-model output of the image.')

    return TrainArgs(**vars(parser.parse_args()))


def main(args: TrainArgs):
    print(f"**** training feature extractor ****")

    def transform_imgs(imgs):
        augmentations = transforms.Compose([
            transforms.RandomCrop(args.resolution) if args.crop else transforms.Resize(args.resolution,
                                                                                       interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(.05, .05, .05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return [augmentations(image.convert("RGB")) for image in imgs]

    if args.use_diffusion_model:
        config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
        model_config = json.loads(config_file.read())
        train_arg_file = open(f"{args.checkpoint_dir}/train_arg_config.json", "r")
        train_arg_config: dict = json.loads(train_arg_file.read())

        model = UNet2DModel(
            **model_config
        )

        model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
        model.eval()
        model.to(args.device)

        noise_kind = train_arg_config.get("noise_kind", "gaussian")
        noise_scheduler_inference = DDIMScheduler(args.train_steps, args.start_at_timestep,
                                                  beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                                  reconstruction_weight=args.reconstruction_weight,
                                                  noise_type=noise_kind)

    def denoise_imgs(batch: torch.Tensor) -> torch.Tensor:
        _, imgs, _, _ = generate_samples(model, noise_scheduler_inference, None, batch, args.eta,
                                         args.steps_to_regenerate, args.start_at_timestep, args.crop, noise_kind)

        return imgs

    data_train = MVTecDataset(args.dataset_path, True, args.item, ["good"],
                              transform_imgs)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    data_test = MVTecDataset(args.dataset_path, False, args.item, ["good"],
                             transform_imgs)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)
    extractor = ResNetFE()
    ae = Autoencoder(extractor)
    ae.init_decoder((3, args.resolution, args.resolution))
    trainer = AETrainer(ae, train_loader, test_loader) if not args.use_diffusion_model else DBTrainer(ae, denoise_imgs,
                                                                                                    train_loader,
                                                                                                    test_loader)
    trainer.train(args.epochs)
    torch.save(extractor.state_dict(), f"{args.save_to}")


if __name__ == '__main__':
    args: TrainArgs = parse_args()
    main(args)
