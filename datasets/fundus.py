"""Fundus few-shot semantic segmentation dataset."""
from __future__ import annotations

import os
from os.path import join

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np


class DatasetFundus(Dataset):
    def __init__(self, datapath: str, shot: int, num: int | None = None, split: str = "test"):
        self.benchmark = "fundus"
        self.shot = shot
        self.split = split
        self.categories = ["1"]
        self.class_ids = range(0, 1)

        root_name = os.path.basename(os.path.normpath(datapath)).lower()
        if root_name == "fundus":
            self.base_path = datapath
        else:
            self.base_path = join(datapath, "Fundus")

        self.img_path = join(self.base_path, self.split, "Original")
        self.ann_path = join(self.base_path, self.split, "Ground truth")

        if not os.path.isdir(self.img_path):
            raise FileNotFoundError(f"Fundus image directory not found: {self.img_path}")
        if not os.path.isdir(self.ann_path):
            raise FileNotFoundError(f"Fundus mask directory not found: {self.ann_path}")

        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.total_samples = len(self.img_metadata_classwise["1"])
        if self.total_samples < 2:
            raise ValueError(
                f"Fundus split '{self.split}' needs at least 2 paired images, "
                f"but found {self.total_samples}."
            )
        self.num = self.total_samples if num is None else num

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> dict:
        tgt_name, ref_names, class_sample = self.sample_episode(idx)
        tgt_img, tgt_mask, ref_imgs, ref_masks = self.load_frame(tgt_name, ref_names)

        batch = {
            "tgt_img": tgt_img,
            "tgt_mask": tgt_mask.float(),
            "ref_imgs": ref_imgs,
            "ref_masks": ref_masks,
            "class_id": torch.tensor(class_sample),
        }
        return batch

    def load_frame(self, tgt_name: str, ref_names: list[str]) -> tuple:
        tgt_img = Image.open(tgt_name).convert("RGB")
        ref_imgs = [Image.open(name).convert("RGB") for name in ref_names]

        tgt_mask = self.read_mask(join(self.ann_path, os.path.basename(tgt_name)))
        ref_masks = [self.read_mask(join(self.ann_path, os.path.basename(name))) for name in ref_names]
        return tgt_img, tgt_mask, ref_imgs, ref_masks

    @staticmethod
    def read_mask(img_name: str) -> torch.Tensor:
        mask = torch.tensor(np.array(Image.open(img_name).convert("L")))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx: int) -> tuple:
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]
        candidates = self.img_metadata_classwise[class_sample]

        tgt_name = np.random.choice(candidates, 1, replace=False)[0]
        ref_names = []
        while len(ref_names) < self.shot:
            ref_name = np.random.choice(candidates, 1, replace=False)[0]
            if tgt_name != ref_name:
                ref_names.append(ref_name)

        return tgt_name, ref_names, class_id

    def build_img_metadata_classwise(self) -> dict:
        img_metadata_classwise = {cat: [] for cat in self.categories}
        img_candidates = sorted(
            join(self.img_path, name) for name in os.listdir(self.img_path) if name.lower().endswith(".png")
        )

        for img_path in img_candidates:
            mask_path = join(self.ann_path, os.path.basename(img_path))
            if os.path.isfile(mask_path):
                img_metadata_classwise["1"].append(img_path)
        return img_metadata_classwise


def build(args) -> DatasetFundus:
    split = getattr(args, "fundus_split", "test")
    num = getattr(args, "fundus_num_episodes", None)
    return DatasetFundus(datapath=args.data_root, shot=args.shots, num=num, split=split)
