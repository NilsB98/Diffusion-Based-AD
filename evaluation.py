# Load model and dataset and run evaluation metrics on those.
import json
from collections import Counter

import torch
from diffusers import UNet2DModel
from torch.utils.data import DataLoader
from torchvision import transforms

from inference_ddim import InferenceArgs, run_inference_step, parse_args
from loader.loader import MVTecDataset
from schedulers.scheduling_ddim import DDIMScheduler
from utils.anomalies import calc_threshold


def eval_diffmap_threshold(args: InferenceArgs):
    # train loop
    print("**** starting eval *****")
    config_file = open(f"{args.checkpoint_dir}/model_config.json", "r")
    model_config = json.loads(config_file.read())
    train_arg_file = open(f"{args.checkpoint_dir}/train_arg_config.json", "r")
    train_arg_config: dict = json.loads(train_arg_file.read())

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
    test_data = MVTecDataset(args.dataset_path, True, args.mvtec_item, args.mvtec_item_states,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

    # set model, optimizer, scheduler
    model = UNet2DModel(
        **model_config
    )

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/{args.checkpoint_name}"))
    model.eval()
    model.to(args.device)

    diffmap_blur = transforms.GaussianBlur(2 * int(4 * 4 + 0.5) + 1, 4)

    with torch.no_grad():
        # validate and generate images
        noise_kind = train_arg_config.get("noise_kind", "gaussian")

        noise_scheduler_inference = DDIMScheduler(args.train_steps, args.start_at_timestep,
                                                  beta_schedule=args.beta_schedule, timestep_spacing="leading",
                                                  reconstruction_weight=args.reconstruction_weight, noise_type=noise_kind)

        pl_counter = Counter()
        fl_counter = Counter()
        for i, (imgs, states) in enumerate(test_loader):
            run_inference_step(diffmap_blur, None, [], i, imgs, model, noise_kind,
                               noise_scheduler_inference, states, None, args.eta, args.num_inference_steps,
                               args.start_at_timestep, args.patch_imgs, args.plt_imgs, args.img_dir, pl_counter, fl_counter)

        threshold_pl = calc_threshold(dict(pl_counter), .999, 1000)
        threshold_fl = calc_threshold(dict(fl_counter), .999, 1000)

    print(f"{threshold_pl=:.4f}")
    print(f"{threshold_fl=:.4f}")


if __name__ == '__main__':
    args: InferenceArgs = parse_args()
    eval_diffmap_threshold(args)
