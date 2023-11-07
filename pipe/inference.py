import os
from pathlib import Path

import torch

from pipeline_reconstruction_ddim import DDIMReconstructionPipeline
from utils import diffmap
from utils.diffmap import create_diffmaps
from utils.metrics import scores_batch
from utils.visualize import add_batch_overlay, plot_single_channel_imgs, plot_rgb_imgs, gray_to_rgb, \
    split_batch_into_patch, unnormalize_original_img, stitch_batch_patches, output_to_img


def run_inference_step(extractor, diffmap_blur, eval_scores, gts, img_file_hints, imgs, model, noise_kind,
                       noise_scheduler_inference, states, writer, eta, num_inference_steps, start_at_timestep,
                       patch_imgs, plt_imgs, img_dir, pl_counter=None, fl_counter=None, smoothing_kernel_size=3, pl_threshold=1., fl_threshold=1., fl_contrib=.7, pl_contrib=.7):
    originals, reconstructions, diffmaps, history = generate_samples(model, noise_scheduler_inference, extractor, imgs, eta,
                                                                     num_inference_steps, start_at_timestep, patch_imgs,
                                                                     noise_kind, smoothing_kernel_size)
    # analysis of thresholds:
    if pl_counter is not None and fl_counter is not None:
        diffmap.count_values(diffmaps['diffmap_pl'], factor=5000, counter=pl_counter)
        diffmap.count_values(diffmaps['diffmap_fl'], factor=5000, counter=fl_counter)

    diffmaps = diffmap.normalize_diffmaps(diffmaps, {'threshold_pl': pl_threshold, 'threshold_fl': fl_threshold})
    anomaly_maps = diffmap.diff_maps_to_anomaly_map(diffmaps, {'diffmap_fl': fl_contrib, 'diffmap_pl': pl_contrib}, diffmap_blur)
    overlays = add_batch_overlay(originals, anomaly_maps)

    if eval_scores is not None:
        eval_scores.update(scores_batch(gts, anomaly_maps))

    for idx in range(len(gts)):
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        # iterate over created diffmaps
        single_channel_imgs = [gts[idx].cpu()]
        titles = ["ground truth"]
        c_maps = ['gray']

        for map_name, map_value in diffmaps.items():
            single_channel_imgs.append(map_value[idx].cpu())
            titles.append(map_name)
            c_maps.append('viridis')

        single_channel_imgs.append(anomaly_maps[idx].cpu())
        titles.append("anomaly-map")
        c_maps.append('gray')
        v_maxs = [1] * len(titles)

        plot_single_channel_imgs(single_channel_imgs, titles, cmaps=c_maps, vmaxs=v_maxs,
                                 save_to=f"{img_dir}/{img_file_hints}_{states[idx]}_{idx}_heatmap.png", show_img=plt_imgs)

        plot_rgb_imgs([originals[idx].cpu(), reconstructions[idx].cpu(), overlays[idx].cpu()], ["original", "reconstructed", "overlay"],
                      save_to=f"{img_dir}/{img_file_hints}_{states[idx]}_{idx}.png", show_img=plt_imgs)

        if writer is not None:
            for t, im in zip(history["timesteps"], history["images"]):
                writer.add_images(f"{img_file_hints}_{states[0]}_process", im[idx].unsqueeze(0), t)

            # TODO add logic for diffmap_fl
            writer.add_images(f"{img_file_hints}_{states[0]}_results (ori, rec, diff, pred, gt)", torch.stack(
                [originals[idx].cpu(), reconstructions[idx].cpu(), gray_to_rgb(diffmaps['diffmap_pl'])[0].cpu(), gray_to_rgb(anomaly_maps[idx].cpu())[0],
                 gray_to_rgb(gts[idx].cpu())[0]]))


def generate_samples(model, noise_scheduler, extractor, original_images, eta, steps_to_regenerate, start_at_timestep,
                     patch_imgs=False, noise_kind='gaussian', fl_smoothing_kernel_size=3):
    num_imgs = len(original_images)
    if patch_imgs:
        original_images = split_batch_into_patch(original_images, model.sample_size)

    pipeline = DDIMReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
        noise_kind=noise_kind
    )

    original_images = original_images.to(model.device)
    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    pipe_output = pipeline(
        batch_size=len(original_images),
        generator=generator,
        num_inference_steps=steps_to_regenerate,
        original_images=original_images,
        eta=eta,
        start_at_timestep=start_at_timestep,
        output_type="torch",
    )
    reconstruction = pipe_output.images
    history = pipe_output.history

    original = unnormalize_original_img(original_images)

    if patch_imgs:
        reconstruction = stitch_batch_patches(reconstruction, num_imgs)
        original = stitch_batch_patches(original, num_imgs)

    diff_maps = create_diffmaps(original, reconstruction, extractor, model.sample_size, fl_smoothing_kernel_size)
    history["images"] = [output_to_img(output, num_imgs) for output in history["images"]]

    return original.cpu(), reconstruction.cpu(), diff_maps, history
