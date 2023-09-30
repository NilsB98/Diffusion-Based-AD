import math
from typing import List, Callable

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet, wide_resnet101_2, Wide_ResNet101_2_Weights
from torch import Tensor, nn
from torchvision import transforms
import torch.nn.functional as F

from loader.loader import MVTecDataset


def get_feature_extractor(path_state_dict=None) -> ResNet:
    model: ResNet = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.DEFAULT)
    model.eval()

    def forward_impl(x: Tensor) -> List:
        # Adjusted the _forward_impl function from the ResNet class to return the hidden layers instead
        # see https://github.com/pytorch/vision/blob/v0.15.2/torchvision/models/resnet.py#L266
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        l1 = model.layer1(x)
        l2 = model.layer2(l1)
        l3 = model.layer3(l2)

        return [l2, l3]

    model._forward_impl = forward_impl

    if path_state_dict is not None:
        model.load_state_dict(torch.load(path_state_dict))
    return model


def compare_features(fe: ResNet, batch1: Tensor, batch2: Tensor) -> Tensor:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    batch1 = normalize(batch1)
    batch2 = normalize(batch2)

    features1 = fe(batch1)
    features2 = fe(batch2)

    diff_maps = torch.zeros((batch1.shape[0], 1, *batch1.shape[2:])).to(batch1.device)

    for i in range(len(features1)):
        d_map = 1 - F.cosine_similarity(mean_conv(features1[i]), mean_conv(features2[i]))
        d_map = torch.unsqueeze(d_map, dim=1)
        d_map = F.interpolate(d_map, size=batch1.shape[-1], mode='bilinear', align_corners=True)
        diff_maps += d_map

    return diff_maps


def fine_tune_fe(fe: ResNet, noise_denoise_pipe: Callable, loader, epochs: int, save_to: str) -> ResNet:
    fe.train()

    for param in fe.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        fe.parameters(),
        weight_decay=1e-6,
        lr=1e-4,
        betas=(0.95, 0.999),
        eps=1e-08,
    )

    def loss_fn(a: List[Tensor], b: List[Tensor]) -> Tensor:
        cos_loss = torch.nn.CosineSimilarity()
        loss = torch.zeros(1).to(a[0].device)
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
        return loss

    for epoch in range(epochs):
        for batch, _ in loader:
            # TODO done differently in DDAD
            # idea here to do domain adaptation: compare two images and increase the cosine similarity between both
            batch = batch.to(torch.device('cuda:0'))
            if batch.shape[0] % 2 != 0:
                batch = batch[1:]
            bs_half = batch.shape[0] // 2

            # reconstructed_imgs = noise_denoise_pipe(batch)

            features_a = fe(batch[:bs_half])
            features_b = fe(batch[bs_half:])

            loss = loss_fn(features_a, features_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())

    torch.save(fe.state_dict(), save_to)

    return fe


def mean_conv(x: Tensor, kernel_size=3):
    in_shape = x.shape
    weights = torch.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    weights = weights.view(1, 1, kernel_size, kernel_size)
    x = x.view(-1, 1, x.shape[2], x.shape[3])
    out = F.conv2d(x, weights, stride=1, padding=kernel_size // 2)

    return out.view(in_shape)


if __name__ == '__main__':
    def transform_imgs(imgs):
        augmentations = transforms.Compose([
            transforms.RandomCrop(256) if True else transforms.Resize(256,
                                                                      interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return [augmentations(image.convert("RGB")) for image in imgs]


    data_train = MVTecDataset("C:/Users/nilsb/Documents/mvtec_anomaly_detection.tar", True, "leather", ["good"],
                              transform_imgs)

    train_loader = DataLoader(data_train, batch_size=8, shuffle=True)
    model = get_feature_extractor()
    model.to(torch.device('cuda:0'))
    model = fine_tune_fe(model, None, train_loader, 3, "checkpoints/feature_extractor/extractor.pt")
