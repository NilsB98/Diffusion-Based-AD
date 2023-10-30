import torch

from noise.simplex import simplexGenerator


def validate_step(model, batch, noise_scheduler, num_train_steps, loss_fn, noise_kind):
    with torch.no_grad():
        model.eval()

        clean_imgs = batch.to(model.device)
        if noise_kind == "gaussian":
            noise = torch.randn(clean_imgs.shape, dtype=clean_imgs.dtype).to(clean_imgs.device)
        elif noise_kind == "simplex":
            noise = simplexGenerator.batch_3d_octaves(clean_imgs.shape, 6, 0.6).to(model.device)
        else:
            raise ValueError(f"unknown noise kind: {noise_kind}")

        timesteps = torch.randint(0, num_train_steps, (batch.shape[0],), device=clean_imgs.device).long()
        noisy_images = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

        prediction = model(noisy_images, timesteps).sample
        loss = loss_fn(prediction, noise)

        return loss.item()
