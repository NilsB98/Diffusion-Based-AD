# imports
import os.path
import time

from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from pipe.train import train_step
from pipe.validate import validate_step
from scheduling_ddad import DDADScheduler
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, get_scheduler, DDIMScheduler
from tqdm import tqdm
from loader.loader import MVTecDataset
from utils.visualize import generate_samples

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
TARGET_DEVICE = "cuda"


def transform_images(imgs):
    augmentations = transforms.Compose(
        [
            transforms.Resize(TARGET_RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if RANDOM_FLIP else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    return [augmentations(image.convert("RGB")) for image in imgs]

def main():
    # -------------      load data      ------------
    data_train = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", True, f"{DATASET_NAME}", ["good"],
                              transform_images)
    train_loader = DataLoader(data_train, batch_size=8, shuffle=True)
    test_data = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", False, f"{DATASET_NAME}", STATES,
                             transform_images)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

    # ----------- set model, optimizer, scheduler -----------------
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

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * EPOCHS),
    )
    loss_fn = torch.nn.MSELoss()

    # additional info/util
    timestamp = str(time.time())[:11]
    writer = SummaryWriter(f'{LOG_DIR}/{RUN_NAME}_{timestamp}')

    # -----------------     train loop   -----------------
    print("**** starting training *****")

    if not os.path.exists(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}"):
        os.makedirs(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}")
        config_file = open(f"{CHECKPOINT_DIR}/{RUN_NAME}_{timestamp}/model_config.json", "w+")
        config_file.write(str(model_args))
        config_file.close()

    for epoch in range(EPOCHS):
        model.train()
        model.to(TARGET_DEVICE)

        progress_bar = tqdm(total=len(train_loader) + len(test_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        running_loss_train = 0

        for btc_num, (batch, label) in enumerate(train_loader):
            loss = train_step(model, batch, noise_scheduler, lr_scheduler, loss_fn, optimizer, NUM_TRAIN_STEPS)

            running_loss_train += loss
            progress_bar.update(1)

        running_loss_test = 0
        with torch.no_grad():
            for btc_num, (batch, label, gt) in enumerate(test_loader):
                loss = validate_step(model, batch, noise_scheduler, NUM_TRAIN_STEPS, loss_fn)

                running_loss_test += loss
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
