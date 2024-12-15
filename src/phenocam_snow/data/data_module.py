from pathlib import Path
from typing import Any, Literal

import lightning
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from phenocam_snow.data.dataset import PhenoCamDataset


class PhenoCamDataModule(lightning.LightningDataModule):  # pragma: no cover
    """LightningDataModule that wraps the PhenoCam image dataset class."""

    def __init__(
        self,
        site_name: str,
        train_dir: str | Path,
        train_labels: str | Path,
        test_dir: str | Path,
        test_labels: str | Path,
        batch_size: int = 16,
    ):
        """
        :param site_name: The name of the target PhenoCam site.
        :type site_name: str
        :param train_dir: The directory containing the training images.
        :type train_dir: str
        :param train_labels: The path to the training labels.
        :type train_labels: str
        :param test_dir: The directory containing the testing images.
        :type test_dir: str
        :param test_labels: The path to the testing labels.
        :type test_labels: str
        :param batch_size: The training batch size, defaults to 16.
        :type batch_size: int
        """
        super().__init__()

        self.site_name = site_name
        self.train_dir = train_dir
        self.train_labels = train_labels
        self.test_dir = test_dir
        self.test_labels = test_labels
        self.batch_size = batch_size

        self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        self.augment = transforms.Compose([self.preprocess, transforms.GaussianBlur(3)])

    def prepare_data(
        self,
        train_download_args: dict[str, Any] | None = None,
        train_label_args: dict[str, Any] | None = None,
        test_download_args: dict[str, Any] | None = None,
        test_label_args: dict[str, Any] | None = None,
    ) -> None:
        """
        :param train_download_args: Arguments for downloading training images.
        :type train_download_args: dict[str, Any] | None
        :param train_label_args: Arguments for labeling training images.
        :type train_label_args: dict[str, Any] | None
        :param test_download_args: Arguments for downloading testing images.
        :type test_download_args: dict[str, Any] | None
        :param test_label_args: Argument for labeling testing images.
        :type test_label_args: dict[str, Any] | None

        :raises ValueError: if download or label arguments do not match what was provided at this instance's
            initialization
        """
        if train_download_args:
            if train_download_args["site_name"] != self.site_name:
                raise ValueError(
                    f"{train_download_args['site_name']} != {self.site_name}"
                )
            if train_download_args["save_to"] != self.train_dir:
                raise ValueError(
                    f"{train_download_args['save_to']} != {self.train_dir}"
                )
            print("Downloading train data")
            download(**train_download_args)
        if train_label_args:
            if train_label_args["img_dir"] != self.train_dir:
                raise ValueError(f"{train_label_args['img_dir']} != {self.train_dir}")
            if train_label_args["save_to"] != self.train_labels:
                raise ValueError(
                    f"{train_label_args['save_to']} != {self.train_labels}"
                )
            print("Labeling train data")
            label_images(**train_label_args)

        if test_download_args:
            if test_download_args["site_name"] != self.site_name:
                raise ValueError(
                    f"{test_download_args['site_name']} != {self.site_name}"
                )
            if test_download_args["save_to"] != self.test_dir:
                raise ValueError(f"{test_download_args['save_to']} != {self.test_dir}")
            print("Downloading test data")
            download(**test_download_args)
        if test_label_args:
            if test_label_args["img_dir"] != self.test_dir:
                raise ValueError(f"{test_label_args['img_dir']} != {self.test_dir}")
            if test_label_args["save_to"] != self.test_labels:
                raise ValueError(f"{test_label_args['save_to']} != {self.test_labels}")
            print("Labeling test data")
            label_images(**test_label_args)

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        """
        :param stage: If the stage if "fit", the training data is split 80/20 into training and validation sets. The
            augmented transformation policy is applied to the images. If the stage is "test", the testing dataset is
            loaded and the standard transformation is applied to the images. By default, all three datasets are loaded.
        :type stage: Literal["fit", "test"] | None

        :raise ValueError: if stage is not one of ["fit", "test", None]
        """
        if stage not in ("fit", "test", None):
            raise ValueError(f"{stage} is not a valid stage")

        if stage in ("fit", None):
            img_dataset = PhenoCamDataset(
                self.train_dir, self.train_labels, transform=self.augment
            )
            train_size = round(len(img_dataset) * 0.8)
            val_size = len(img_dataset) - train_size
            self.img_train, self.img_val = random_split(
                img_dataset, [train_size, val_size]
            )
            self.dims = self.img_train[0][0].shape

        if stage in ("test", None):
            self.img_test = PhenoCamDataset(
                self.test_dir, self.test_labels, transform=self.preprocess
            )
            self.dims = getattr(self, "dims", self.img_test[0][0].shape)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.img_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.img_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def get_categories(self) -> list[str]:
        """Gets a list of the image categories, ordered according to their integer encoding.

        :return: A list of categories.
        :rtype: list[str]

        :raises ValueError: if the training categories and testing categories are not equivalent
        """

        def parse_labels_file(labels_path: str | Path) -> list[str]:
            categories = []
            with open(labels_path, "r") as f:
                start_reading = False
                for line in f:
                    if start_reading:
                        if line[0] != "#":
                            break
                        else:
                            _, str_label = line[1:].split(". ")
                            str_label = str_label.strip()
                            categories.append(str_label)
                    if line == "# Categories:\n":
                        start_reading = True

        train_categories = parse_labels_file(self.train_labels)
        test_categories = parse_labels_file(self.test_labels)
        if train_categories != test_categories:
            raise ValueError("train categories do not match test categories")

        return train_categories
