import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import ResNet18_Weights

from .utils import *


class PhenoCamImageDataset(Dataset):
    """PyTorch dataset for PhenoCam images."""

    def __init__(self, img_dir, labels_file, transform=None):
        r"""
        .. highlight:: python

        :param img_dir: The directory where all the images are contained.
        :type img_dir: str
        :param labels_file: The path of the labels file for the images in
            :python:`img_dir`.
        :type labels_file: str
        :param transform: The transform to apply to the images.
        :type transform: torch.nn.Module|None
        """
        df = read_labels(labels_file)
        self.img_labels = df[["img_name", "int_label"]].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        return img, label


class PhenoCamDataModule(LightningDataModule):
    """pytorch_lightning DataModule that wraps the PhenoCam image dataset class."""

    def __init__(
        self, site_name, train_dir, train_labels, test_dir, test_labels, batch_size=16
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
        train_download_args=None,
        train_label_args=None,
        test_download_args=None,
        test_label_args=None,
    ):
        """
        :param train_download_args: Arguments for downloading training images.
        :type train_download_args: Dict[Any]|None
        :param train_label_args: Arguments for labeling training imags. Needs
            to have a key called `"method"` which has two valid values:
            `"in noteobook"` and `"via subdir"`. If the value is the former,
            then the other keys should match the arguments required by
            `utils.label_images_in_notebook`. Otherwise, the other keys should
            match the arguments required by `utils.label_images_via_subdir`.
        :type train_label_args: Dict[Any]|None
        :param test_download_args: Arguments for downloading testing images.
        :type test_download_args: Dict[Any]|None
        :param test_label_args: Argument for labeling testing imagse. See the
            description for `train_label_args`.
        :type test_label_args: Dict[Any]|None
        """
        # Download and/or label train data
        if train_download_args:
            assert train_download_args["site_name"] == self.site_name
            assert train_download_args["save_to"] == self.train_dir
            print("Downloading train data")
            download(**train_download_args)
        if train_label_args:
            assert train_label_args["img_dir"] == self.train_dir
            assert train_label_args["save_to"] == self.train_labels
            print("Labeling train data")
            if train_label_args["method"] == "via subdir":
                del train_label_args["method"]
                label_images_via_subdir(**train_label_args)

        # Download and/or label test data
        if test_download_args:
            assert test_download_args["site_name"] == self.site_name
            assert test_download_args["save_to"] == self.test_dir
            print("Downloading test data")
            download(**test_download_args)
        if test_label_args:
            assert test_label_args["img_dir"] == self.test_dir
            assert test_label_args["save_to"] == self.test_labels
            print("Labeling test data")
            if test_label_args["method"] == "via subdir":
                del test_label_args["method"]
                label_images_via_subdir(**test_label_args)

    def setup(self, stage=None):
        """
        :param stage: If the stage if "fit", the training data is split 70/30
            into training and validation sets. The augmented transformation
            policy is applied to the images. If the stage is "test", the
            testing dataset is loaded and the standard transformation is
            applied to the images. By default, all three datasets are loaded.
        :type stage: str|None
        """
        if stage in ("fit", None):
            img_dataset = PhenoCamImageDataset(
                self.train_dir, self.train_labels, transform=self.augment
            )
            train_size = round(len(img_dataset) * 0.8)
            val_size = len(img_dataset) - train_size
            self.img_train, self.img_val = random_split(
                img_dataset, [train_size, val_size]
            )
            self.dims = self.img_train[0][0].shape
        if stage in ("test", None):
            self.img_test = PhenoCamImageDataset(
                self.test_dir, self.test_labels, transform=self.preprocess
            )
            self.dims = getattr(self, "dims", self.img_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.img_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.img_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.img_test, batch_size=self.batch_size)

    def get_categories(self):
        """Gets a list of the image categories, ordered according to their
           integer encoding.

        :return: A list of categories.
        :rtype: List[str]
        """
        train_categories = []
        with open(self.train_labels, "r") as f:
            start_reading = False
            for line in f:
                if start_reading:
                    if line[0] != "#":
                        break
                    else:
                        _, str_label = line[1:].split(". ")
                        str_label = str_label.strip()
                        train_categories.append(str_label)
                if line == "# Categories:\n":
                    start_reading = True

        test_categories = []
        with open(self.test_labels, "r") as f:
            start_reading = False
            for line in f:
                if start_reading:
                    if line[0] != "#":
                        break
                    else:
                        _, str_label = line[1:].split(". ")
                        str_label = str_label.strip()
                        test_categories.append(str_label)
                if line == "# Categories:\n":
                    start_reading = True

        assert train_categories == test_categories
        return train_categories
