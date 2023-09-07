import torch
from noise.perlin import simplexGenerator


def train_step(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, num_train_steps, noise_kind='normal'):
    assert noise_kind in ['normal', 'perlin']

    model.train()
    clean_imgs = batch.to(model.device)

    noise = None
    if noise_kind == 'normal':
        noise = torch.randn(clean_imgs.shape, dtype=clean_imgs.dtype)
    elif noise_kind == 'perlin':
        # noise = perlin_2d_batch(clean_imgs.shape, (8, 8))
        noise = simplexGenerator.batch_3d_octaves(clean_imgs.shape, 6, 0.6)
    noise = noise.to(clean_imgs.device)

    timesteps = torch.randint(0, num_train_steps, (batch.shape[0],), device=clean_imgs.device).long()
    noisy_images = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

    optimizer.zero_grad()

    prediction = model(noisy_images, timesteps).sample

    loss = loss_fn(prediction, noise)
    loss.backward()

    optimizer.step()
    lr_scheduler.step()

    return loss.item()
