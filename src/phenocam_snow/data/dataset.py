from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import decode_image


class PhenoCamDataset(Dataset):
    """PyTorch dataset for PhenoCam images."""

    def __init__(
        self,
        img_dir: str | Path,
        labels_file: str | Path,
        transform: nn.Module | None = None,
    ):
        r"""
        .. highlight:: python

        :param img_dir: The directory where all the images are contained.
        :type img_dir: str | Path
        :param labels_file: The path of the labels file for the images in :python:`img_dir`.
        :type labels_file: str | Path
        :param transform: The transform to apply to the images.
        :type transform: torch.nn.Module | None
        """
        df = PhenoCamDataset.read_labels(labels_file)
        self.img_labels = df[["filename", "int_label"]]
        if not isinstance(img_dir, Path):
            img_dir = Path(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        img = decode_image(self.img_dir / self.img_labels.iat[idx, 0])
        if self.transform:
            img = self.transform(img)
        label = self.img_labels.iat[idx, 1]
        return img, label

    @staticmethod
    def read_labels(labels_file: str | Path) -> pd.DataFrame:
        """Reads image-label pairs.

        :param labels_file: The path to the labels file.
        :type labels_file: str | Path

        :return: A pandas DataFrame with "filename" (str), "label" (str), and "int_label" (int) columns.
        :rtype: pd.DataFrame
        """
        labels_dict = {}
        with open(labels_file, "r") as f:
            start_reading = False
            for line in f:
                if start_reading:
                    if line[0] != "#":
                        break
                    else:
                        int_label, str_label = line[1:].split(". ")
                        int_label = int(int_label)
                        str_label = str_label.strip()
                        labels_dict[str_label] = int_label
                if line == "# Categories:\n":
                    start_reading = True

        df = pd.read_csv(labels_file, comment="#")
        df["int_label"] = [labels_dict[x] for x in df["label"]]

        return df
