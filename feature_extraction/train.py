import argparse
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from feature_extraction.autoencoder import Autoencoder, AETrainer
from feature_extraction.extractor import ResNetFE
from loader.loader import MVTecDataset


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
    noise_kind: str
    crop: bool
    plt_imgs: bool
    img_dir: str
    calc_val_loss: bool


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
    parser.add_argument('--noise_kind', type=str, default="gaussian",
                        choices=["simplex", "gaussian"],
                        help='Kind of noise to use for the noising steps.')
    parser.add_argument('--crop', action='store_true',
                        help='If set: the image will be cropped to the resolution instead of resized.')
    parser.add_argument('--plt_imgs', action='store_true',
                        help='If set: plot the images with matplotlib')
    parser.add_argument('--calc_val_loss', action='store_true',
                        help='If set: calculate not only the train loss, but also the validation loss during each epoch')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Directory to store the images created during the run. A new directory with the run-id will be created in this directory. If not used images wont be stored except for tensorboard.')

    return TrainArgs(**vars(parser.parse_args()))


if __name__ == '__main__':
    args: TrainArgs = parse_args()
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


    data_train = MVTecDataset(args.dataset_path, True, args.mvtec_item, ["good"],
                              transform_imgs)

    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    data_test = MVTecDataset(args.dataset_path, False, args.mvtec_item, ["good"],
                             transform_imgs)

    test_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)

    extractor = ResNetFE()
    ae = Autoencoder(extractor)
    ae.init_decoder((3, args.resolution, args.resolution))
    trainer = AETrainer(ae, train_loader, test_loader)
    trainer.train(args.epochs)

    torch.save(extractor.state_dict(), f"{args.checkpoint_dir}/{args.run_name}.pt")
