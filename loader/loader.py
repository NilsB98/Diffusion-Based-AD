import os
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, path: str, train: bool, piece: str, states: List[str], transform:callable = None):
        self.transform = transform
        self.train = train
        self.imgs, self.ground_truths, self.obj_states, self.original_imgs = self._load_data(path, train, piece, states)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.train:
            return self.imgs[idx], self.obj_states[idx]
        return self.imgs[idx], self.obj_states[idx], self.ground_truths[idx]

    def _load_data(self, path: str, train: bool, piece: str, states: List[str]):

        assert len(states) > 0, 'at least one state for the pieces should be given.'
        assert states[0] == 'good' if train else True, 'for trainset only the state "good" exists'

        all_img_paths = []
        all_gt_paths = []
        all_states = []

        root_path = Path(path, piece)
        objective_path = root_path
        gt_path = None

        if train:
            objective_path /= "train"
        else:
            objective_path /= "test"
            gt_path = root_path / "ground_truth"

        if states[0] == "all":
            states = os.listdir(objective_path)
        for state in states:
            state_path = objective_path / state
            paths = [path for path in state_path.rglob("*.png")]
            all_img_paths.extend(paths)
            all_states.extend([state for _ in range(len(paths))])

            if not train:
                if state == 'good':
                    all_gt_paths.extend([None for _ in range(len(paths))])
                else:
                    gt_paths = [p for p in (gt_path / state).rglob("*.png")]
                    all_gt_paths.extend(gt_paths)

        piece_imgs = [Image.open(img) for img in all_img_paths]
        tranformed_imgs:List[torch.Tensor] = self.transform(piece_imgs) if self.transform is not None else piece_imgs
        gt_images = [Image.open(img) if img is not None else Image.new('L', tranformed_imgs[0].shape[1:]) for img in all_gt_paths]
        gt_images = self._transform_gt(gt_images, tranformed_imgs[0].shape[1])
        return tranformed_imgs, gt_images, all_states, piece_imgs

    def _transform_gt(self, imgs, target_size):
        augmentations = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ]
        )

        return [augmentations(image) if image is not None else torch.zeros((3, target_size, target_size)) for image in imgs]

