# imports
import argparse
import json
from collections import Counter
from dataclasses import dataclass

import torch
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import feature_extraction
from pipe.inference import run_inference_step
from loader.loader import MVTecDataset
from schedulers.scheduling_ddim import DDIMScheduler
from utils.files import save_args
from utils.metrics import aggregate_img_scores, normalize_pxl_scores


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
    eta: float
    device: str
    dataset_path: str
    shuffle: bool
    img_dir: str
    plt_imgs: bool
    patch_imgs: bool
    batch_size: int
    extractor_path: str
    feature_smoothing_kernel: int
    fl_threshold: float
    pl_threshold: float
    fl_contrib:float
    pl_contrib:float

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
    parser.add_argument('--extractor_path', type=str,
                        help='Path to the feature extractor. This is extractor is used to calculate differences between original and reconstructed images in a deep learning fashion.')
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
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of images to process per batch')
    parser.add_argument('-fsk', '--feature_smoothing_kernel', type=int, default=3,
                        help='Size of the kernel to be used for smoothing the extracted features. Set to 1 for no smoothing.')
    parser.add_argument('--pl_threshold', type=float, default=0.029,
                        help='Pixel level threshold for difference-map')
    parser.add_argument('--fl_threshold', type=float, default=0.33,
                        help='Feature level threshold for difference-map')
    parser.add_argument('--fl_contrib', type=float, default=0.7,
                        help='Contribution of the feature-level diffMap to the combined diffMap')
    parser.add_argument('--pl_contrib', type=float, default=0.7,
                        help='Contribution of the pixel-level diffMap to the combined diffMap')

    return InferenceArgs(**vars(parser.parse_args()))


def main(args: InferenceArgs, writer: SummaryWriter):
    # train loop
    print("**** starting inference *****")
    config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
    model_config = json.loads(config_file.read())
    train_arg_file = open(f"{args.checkpoint_dir}/train_arg_config.json", "r")
    train_arg_config: dict = json.loads(train_arg_file.read())
    save_args(args, args.img_dir, "inference_args")

    augmentations = transforms.Compose(
        [
            transforms.Resize(model_config["sample_size"],
                              interpolation=transforms.InterpolationMode.BILINEAR) if not args.patch_imgs else transforms.Lambda(
                lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(imgs):
        return [augmentations(image.convert("RGB")) for image in imgs]

    # data loader
    test_data = MVTecDataset(args.dataset_path, False, args.mvtec_item, args.mvtec_item_states,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

    # set model, optimizer, scheduler
    model = UNet2DModel(
        **model_config
    )

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
    model.eval()
    model.to(args.device)

    extractor = feature_extraction.ResNetFE(args.extractor_path)
    extractor.eval()
    extractor.to(args.device)

    diffmap_blur = transforms.GaussianBlur(2 * int(4 * 4 + 0.5) + 1, 4)

    with torch.no_grad():
        # validate and generate images
        noise_kind = train_arg_config.get("noise_kind", "gaussian")
        noise_scheduler_inference = DDIMScheduler(args.train_steps, args.start_at_timestep,
                                                  beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                                  reconstruction_weight=args.reconstruction_weight, noise_type=noise_kind)
        eval_scores = Counter()

        for i, (imgs, states, gts) in enumerate(test_loader):
            run_inference_step(extractor, diffmap_blur, eval_scores, gts, i, imgs, model, noise_kind,
                               noise_scheduler_inference, states, writer, args.eta, args.num_inference_steps,
                               args.start_at_timestep, args.patch_imgs, args.plt_imgs, args.img_dir,
                               smoothing_kernel_size=args.feature_smoothing_kernel, pl_threshold=args.pl_threshold, fl_threshold=args.fl_threshold)

        normalize_pxl_scores(len(test_loader), eval_scores)
        aggregate_img_scores(eval_scores)

        writer.add_hparams({'category': args.mvtec_item, 'eta': args.eta,
                            'recon_weight': args.reconstruction_weight, 'states': ','.join(args.mvtec_item_states),
                            't': args.start_at_timestep, 'num_steps': args.num_inference_steps,
                            'input_size': model_config["sample_size"], 'patching': args.patch_imgs}, dict(eval_scores),
                           run_name=f'hp')
        print(eval_scores)


if __name__ == '__main__':
    args: InferenceArgs = parse_args()
    writer = SummaryWriter(f'{args.log_dir}/{args.checkpoint_dir.split("/")[-1]}') if args.log_dir else None
    main(args, writer)
    writer.flush()
    writer.close()