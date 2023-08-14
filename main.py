# imports
import os.path
import time

from torch.utils.data import Subset, DataLoader
from datasets import load_dataset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from scheduling_ddad import DDADScheduler
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, get_scheduler, DDIMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from pipeline_reconstruction import ReconstructionPipeline
from loader.loader import MVTecDataset

# dataset
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
RUN_NAME = "hazelnut_128res"
DATASET_NAME = "hazelnut"
STATES = ["cut"]
TARGET_RESOLUTION = 128
EPOCHS = 25
NUM_TRAIN_STEPS, BETA_SCHEDULE = 1000, "linear"
RANDOM_FLIP = False
SAVE_N_EPOCH = 10

timestamp = str(time.time())[:11]
writer = SummaryWriter(f'{LOG_DIR}/{RUN_NAME}_{timestamp}')

augmentations = transforms.Compose(
    [
        transforms.Resize(TARGET_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform_images(imgs):
    return [augmentations(image.convert("RGB")) for image in imgs]


# data loader
data_train = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", True, f"{DATASET_NAME}", ["good"],
                          transform_images)
train_loader = DataLoader(data_train, batch_size=8, shuffle=True)
test_data = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", False, f"{DATASET_NAME}", STATES,
                         transform_images)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

# set model, optimizer, scheduler
model_args = {
    "sample_size": TARGET_RESOLUTION,
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
noise_scheduler = DDPMScheduler(NUM_TRAIN_STEPS, beta_schedule=BETA_SCHEDULE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    weight_decay=1e-6,
    lr=1e-4,
    betas=(0.95, 0.999),
    eps=1e-08,
)

# Initialize the learning rate scheduler
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(train_loader) * EPOCHS),
)


def generate_samples(model, noise_scheduler, plt_title, original_images):
    pipeline = ReconstructionPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        generator=generator,
        num_inference_steps=1000,
        output_type="numpy",
        original_images=original_images.to(model.device)
    ).images

    images_processed = (images * 255).round().astype("uint8")
    images = torch.from_numpy(images_processed)
    images = torch.permute(images, (0, 3, 1, 2))

    original_images = transforms.Normalize([-0.5 * 2], [2])(original_images)
    originals = (original_images * 255).round().type(torch.uint8)

    grid = make_grid(torch.cat((images, originals), 0), 4)
    show(grid, plt_title)
    return grid


def show(imgs, title):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.title(title)
    plt.show()


def main():
    # train loop
    print("**** starting training *****")
    loss_fn = torch.nn.MSELoss()
    # check if checkpoint dir exist
    if not os.path.exists(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}"):
        os.makedirs(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}")
        config_file = open(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}/model_config.json", "w+")
        config_file.write(str(model_args))
        config_file.close()

    for epoch in range(EPOCHS):
        model.train()
        model.to('cuda')

        progress_bar = tqdm(total=len(train_loader) + len(test_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss_train = 0

        for btc_num, (batch, label) in enumerate(train_loader):
            clean_imgs = batch
            clean_imgs = clean_imgs.to('cuda')
            noise = torch.randn(clean_imgs.shape, dtype=clean_imgs.dtype).to(clean_imgs.device)

            timesteps = torch.randint(0, NUM_TRAIN_STEPS, (batch.shape[0],), device=clean_imgs.device).long()
            noisy_images = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

            optimizer.zero_grad()

            prediction = model(noisy_images, timesteps).sample

            loss = loss_fn(prediction, noise)
            loss.backward()
            running_loss_train += loss.item()

            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)

        running_loss_test = 0
        with torch.no_grad():

            # run validation
            for btc_num, (batch, label, gt) in enumerate(test_loader):
                clean_imgs = batch.to("cuda")
                noise = torch.randn(clean_imgs.shape, dtype=clean_imgs.dtype).to(clean_imgs.device)
                timesteps = torch.randint(0, NUM_TRAIN_STEPS, (batch.shape[0],), device=clean_imgs.device).long()
                noisy_images = noise_scheduler.add_noise(clean_imgs, noise, timesteps)

                prediction = model(noisy_images, timesteps).sample
                loss = loss_fn(prediction, noise)
                running_loss_test += loss.item()
                progress_bar.update(1)

            progress_bar.set_postfix_str(
                f"Train Loss: {running_loss_train / len(train_loader)}, Test Loss: {running_loss_test / len(test_loader)}")
            progress_bar.close()

            if epoch % 100 == 0:
                # generate images
                noise_scheduler_inference = DDADScheduler(NUM_TRAIN_STEPS, beta_schedule=BETA_SCHEDULE)
                train_grid = generate_samples(model, noise_scheduler_inference, f"Train samples {epoch=}",
                                              next(iter(train_loader))[0])
                test_grid = generate_samples(model, noise_scheduler_inference, f"Test samples {epoch=}",
                                             next(iter(test_loader))[0])

                writer.add_image(f'Test samples {epoch=}', test_grid, epoch)

            if epoch % SAVE_N_EPOCH == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}/epoch_{epoch}.pt")

        writer.add_scalar('Loss/train', running_loss_train, epoch)
        writer.add_scalar('Loss/test', running_loss_test, epoch)

    writer.add_hparams({'lr': -1, 'category': DATASET_NAME, 'defects': str(STATES)}, {'MSE': running_loss_test},
                       run_name='hp')
    writer.flush()
    writer.close()

    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}/epoch_{EPOCHS}.pt")


if __name__ == '__main__':
    main()
