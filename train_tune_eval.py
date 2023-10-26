"""
Meta Script to run the training, parameter tuning and evaluation in one script.
"""
import argparse
import os
from dataclasses import dataclass
from typing import List

from torch.utils.tensorboard import SummaryWriter

import evaluation
import inference_ddim
import train_ddim
import train_extractor


@dataclass
class MetaArgs:
    # set phases of the script
    skip_train: bool
    skip_extractor: bool
    skip_threshold: bool
    skip_evaluation: bool

    # configure paths
    diffusion_checkpoint_dir: str
    diffusion_checkpoint_name: str
    extractor_path: str
    log_dir: str
    img_dir: str

    # diffusion training configuration
    flip: bool
    rotate: float
    color_jitter: float
    epochs_diffusion: int
    train_steps: int
    save_n_epochs: int
    calc_val_loss: bool

    # diffusion inference config
    steps_to_regenerate: int
    start_at_timestep: int
    shuffle: bool

    # diffusion config
    beta_schedule: str
    reconstruction_weight: float
    eta: float
    noise_kind: str

    # extractor config
    epochs_extractor: int
    train_extractor_on_diff_model: bool
    feature_smoothing_kernel: int

    # data config
    dataset_path: str
    batch_size: int
    item: str
    item_states: list

    # custom thresholds
    pxl_threshold: float
    feature_threshold: float

    # general config
    resolution: int
    use_patching_approach: bool
    device: str
    plt_imgs: bool
    run_id: str

    # TODO implement
    pl_diffmap_influence: float
    fl_diffmap_influence: float


def main():
    # get args
    diffusion_train_args, extractor_train_args, eval_args, all_args = parse_args()

    # setup the logger
    writer = SummaryWriter(all_args.log_dir, flush_secs=30)

    # train the diffusion model
    if not all_args.skip_train:
        train_ddim.main(diffusion_train_args)   # TODO add writer

    # train the extractor
    if not all_args.skip_extractor:
        train_extractor.main(extractor_train_args)  # TODO add writer

    # find the pixel- and feature-level thresholds for the difference-map
    if not all_args.skip_threshold:
        thresholds = evaluation.eval_diffmap_threshold(eval_args)   # TODO add writer
        eval_args.pl_threshold = thresholds['threshold_pl']
        eval_args.fl_threshold = thresholds['threshold_fl']

    # evaluate the model
    inference_ddim.main(eval_args, writer)


def parse_args():
    parser = argparse.ArgumentParser(description='Add config for the train-tune-eval pipeline')
    # set phases of the script
    parser.add_argument('--skip_train', action='store_true',
                        help='Train the diffusion model')
    parser.add_argument('--skip_extractor', action='store_true',
                        help='Train the feature extractor')
    parser.add_argument('--skip_threshold', action='store_true',
                        help='Evaluate the thresholds for the diff-maps automatically based on the training data. '
                             'These thresholds are used to determine whether a pixel is anomalous or not.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Evaluate the models and run metrics.')

    # configure paths
    parser.add_argument('--diffusion_checkpoint_dir', type=str, required=True,
                        help='directory path to store/load the checkpoint of the diffusion model')
    parser.add_argument('--diffusion_checkpoint_name', type=str, required=True,
                        help='Name of the diffusion models checkpoint (i.e. the .pt file)')
    parser.add_argument('--extractor_path', type=str, required=True,
                        help='Full path of the extractor checkpoint. Will either be created or'
                             ' loaded depending on "skip_extractor flag".')
    parser.add_argument('--log_dir', type=str, default="logs",
                        help='directory path to store logs')
    parser.add_argument('--img_dir', type=str, default="generated_imgs",
                        help='directory path to store generated images')

    # diffusion training configuration
    parser.add_argument('--flip', action='store_true',
                        help='Flip the training images during training.')
    parser.add_argument('--rotate', type=float, default=0,
                        help='degree of rotation to augment training data with')
    parser.add_argument('--color_jitter', type=float, default=0,
                        help='amount of color jitter to augment training data with')
    parser.add_argument('--epochs_diffusion', type=int, default=1000,
                        help='Number of epochs to train the diffusion model for.')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='Number of steps the diffusion model should make for a full diffusion pass.')
    parser.add_argument('--save_n_epochs', type=int, default=50,
                        help='Number of epochs after which a checkpoint should be saved during training of '
                             'the diffusion model.')
    parser.add_argument('--calc_val_loss', action='store_true',
                        help='Calculate the validation loss during the training of the diffusion model.')

    # diffusion config
    parser.add_argument('--beta_schedule', type=str, default="linear",
                        help='Type of schedule for the beta/variance values')
    parser.add_argument('--recon_weight', type=float, default=1, dest="reconstruction_weight",
                        help='Influence of the original sample during generation')
    parser.add_argument('--eta', type=float, default=0,
                        help='Stochasticity parameter of DDIM, with eta=1 being DDPM and eta=0 meaning no randomness')
    parser.add_argument('--noise_kind', type=str, default="gaussian", choices=['gaussian', 'simplex'],
                        help='Kind of noise to use for the diffusion model.')

    # diffusion inference config
    parser.add_argument('--steps_to_regenerate', type=int, default=25,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--start_at_timestep', type=int, default=250,
                        help='At which timestep/how many timesteps should be regenerated')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the data during the evaluation')

    # extractor config
    parser.add_argument('--epochs_extractor', type=int, default=30,
                        help='Number of epochs to train the extractor.')
    parser.add_argument('--extractor_use_diff', action='store_true', dest='train_extractor_on_diff_model',
                        help='Train the extractor on data generated by the diffusion model.')
    parser.add_argument('-fsk', '--feature_smoothing_kernel', type=int, default=3,
                        help='Size of the kernel to be used for smoothing the extracted features. Set to 1 for no smoothing.')

    # data config
    parser.add_argument('--item', type=str, required=True,
                        help='name of the item within the Dataset to train on')
    parser.add_argument('--item_states', type=list, nargs="+", default=["all"],
                        help="States of the  items that should be used. Available options depend on the selected item. Set to 'all' to include all states")
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images to process per batch')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='directory path to the dataset')

    # custom thresholds:
    parser.add_argument('--pxl_threshold', type=float, default=0.01,
                        help='Threshold at which a pixel should be evaluated as anomalous in the pxl-diffmap.')
    parser.add_argument('--feature_threshold', type=float, default=0.4,
                        help='Threshold at which a pixel should be evaluated as anomalous in the feature-diffmap.')


    # general config
    parser.add_argument('--device', type=str, default="cuda",
                        help='device to train on')
    parser.add_argument('--plt_imgs', action='store_true',
                        help='Plot the images with matplot lib. I.e. call plt.show()')
    parser.add_argument('--patching', action='store_true', dest='use_patching_approach',
                        help='If the image size is larger than the models input, split input into multiple patches and stitch it together afterwards.')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the images. Either resized or cropped.')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Id of the run. The created checkpoint- and log directory will be named like this.')

    # diffmap config
    parser.add_argument('--pl_diffmap_influence', type=float, default=0.7,
                        help='Contribution of the pixel-level diffmap to the combined diffmap')
    parser.add_argument('--fl_diffmap_influence', type=float, default=0.7,
                        help='Contribution of the feature-level diffmap to the combined diffmap')

    args = MetaArgs(**vars(parser.parse_args()))

    diffusionTrainArgs = train_ddim.TrainArgs(args.diffusion_checkpoint_dir, args.run_id, args.item, args.flip, args.rotate, args.color_jitter, args.resolution, args.epochs_diffusion, args.save_n_epochs, args.dataset_path, args.train_steps, args.beta_schedule, args.device, args.reconstruction_weight, args.eta, args.batch_size, args.noise_kind, args.use_patching_approach, args.log_dir, args.img_dir, args.plt_imgs, args.calc_val_loss, args.extractor_path, args.diffusion_checkpoint_name)
    checkpoint_dir = os.path.join(args.diffusion_checkpoint_dir, args.run_id)

    extractorTrainArgs = train_extractor.TrainArgs(checkpoint_dir, args.item, args.flip, args.resolution, args.epochs_extractor, args.dataset_path, args.train_steps, args.beta_schedule, args.device, args.reconstruction_weight, args.eta, args.batch_size, args.noise_kind, args.use_patching_approach, args.diffusion_checkpoint_name, args.extractor_path, args.start_at_timestep, args.steps_to_regenerate, args.train_extractor_on_diff_model)

    evalArgs = inference_ddim.InferenceArgs(args.steps_to_regenerate, args.start_at_timestep, args.reconstruction_weight, args.item, args.item_states, checkpoint_dir, args.diffusion_checkpoint_name, args.log_dir, args.train_steps, args.beta_schedule, args.eta, args.device, args.dataset_path, args.shuffle, args.img_dir, args.plt_imgs, args.use_patching_approach, args.batch_size, args.extractor_path, args.feature_smoothing_kernel, args.feature_threshold, args.pxl_threshold)

    return diffusionTrainArgs, extractorTrainArgs, evalArgs, args


if __name__ == '__main__':
    main()
