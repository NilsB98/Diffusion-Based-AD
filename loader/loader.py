import os
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm


class MVTecDataset(Dataset):
    def __init__(self, path: str, train: bool, piece: str, states: List[str], transform:callable = None):
        self.transform = transform
        self.train = train
        self.all_gt_paths, self.all_obj_states, self.all_img_paths = self._load_data(path, train, piece, states)

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.all_img_paths[idx])
        img = self.transform([img])[0]
        state = self.all_obj_states[idx]

        if self.train:
            return img, state

        gt = Image.open(self.all_gt_paths[idx]) if self.all_gt_paths[idx] is not None else Image.new('L', img.shape[1:])
        gt = self._transform_gt([gt], img.shape[1], img.shape[0])
        return img, state, gt


    def _load_data(self, path: str, train: bool, piece: str, states: List[str]):

        assert len(states) > 0, 'at least one state for the pieces should be given.'
        assert states[0] == 'good' if train else True, 'for trainset only the state "good" exists'

        all_img_paths = []
        all_gt_paths = []
        all_states = []

        root_path = Path(path, piece)
        objective_path = root_path
        gt_path = None

        assert root_path.exists(), f"'{root_path}' doesn't exists. Please make sure that @path='{path}' points to the dataset directory and @piece='{piece}' is the directory of the object you want to use."

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

        self.all_states = all_states
        self.all_gt_paths = all_gt_paths
        self.all_img_paths = all_img_paths
        return all_gt_paths, all_states, all_img_paths

    def _transform_gt(self, imgs, target_size_h, target_size_w):
        augmentations = transforms.Compose(
            [
                transforms.Resize((target_size_h, target_size_w), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ]
        )

        return [augmentations(image) if image is not None else torch.zeros((3, target_size_h, target_size_w)) for image in imgs]

