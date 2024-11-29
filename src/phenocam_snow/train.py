from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from phenocam_snow.data import PhenoCamDataModule
from phenocam_snow.model import PhenoCamResNet


def main():
    parser = ArgumentParser(
        description="Train a model to classify images from a given PhenoCam site"
    )
    parser.add_argument("site_name")
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        default="resnet18",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--new",
        action="store_true",
        default=False,
        help="If given, trains and tests on new data. --n_train, --n_test, --classes are required.",
    )
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument(
        "--existing",
        action="store_true",
        default=False,
        help="If given, trains and tests on existing data. --train_dir, --test_dir are required",
    )
    parser.add_argument(
        "--train_dir", help="The file path of the train images directory."
    )
    parser.add_argument(
        "--test_dir", help="The file path of the test images directory."
    )
    parser.add_argument("--classes", nargs="+", help="The image classes to use.")
    args = parser.parse_args()

    label_method = "via subdir"  # can't do "in notebook" from a script
    if args.new and args.existing:
        print("Cannot specify both --new and --existing")
    elif args.new:
        train_model_with_new_data(
            args.model,
            args.learning_rate,
            args.weight_decay,
            args.site_name,
            label_method,
            args.n_train,
            args.n_test,
            args.classes,
        )
    elif args.existing:
        train_model_with_existing_data(
            args.model,
            args.learning_rate,
            args.weight_decay,
            args.site_name,
            args.train_dir,
            args.test_dir,
            args.classes,
        )
    else:
        print("Please specify either --new or --existing")


def train_model_with_new_data(
    resnet_variant: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ],
    learning_rate: float,
    weight_decay: float,
    site_name: str,
    label_method: str,
    n_train: int,
    n_test: int,
    classes: list[str],
) -> PhenoCamResNet:
    """Pipeline for building a model on new data.

    :param resnet_variant: The ResNet variant to use as the model backbone.
    :type resnet_variant: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    :param learning_rate: The learning rate to use.
    :type learning_rate: float
    :param weight_decay: The weight decay to use.
    :type weight_decay: float
    :param site_name: The name of the PhenoCam site you want.
    :type site_name: str
    :param label_method: How you wish to label images ("in notebook" or "via subdir").
    :type label_method: str
    :param n_train: The number of training images to use.
    :type n_train: int
    :param n_test: The number of testing images to use.
    :type n_test: int
    :param classes: The image classes.
    :type classes: list[str]

    :return: The best model obtained during training.
    :rtype: PhenoCamResNet
    """
    valid_label_methods = ["in notebook", "via subdir"]
    if label_method not in valid_label_methods:
        raise ValueError(f"{label_method} is not a valid label method")

    train_dir = f"{site_name}_train"
    test_dir = f"{site_name}_test"
    train_labels = f"{train_dir}/labels.csv"
    test_labels = f"{test_dir}/labels.csv"

    data_module = PhenoCamDataModule(
        site_name, train_dir, train_labels, test_dir, test_labels
    )
    base_download_args = {"site_name": site_name}
    base_label_args = {
        "site_name": site_name,
        "categories": classes,
        "method": label_method,
    }
    data_module.prepare_data(
        train_download_args=base_download_args
        + {"save_to": train_dir, "n_photos": n_train},
        train_label_args=base_label_args
        + {"img_dir": train_dir, "save_to": train_labels},
        test_download_args=base_download_args
        + {"save_to": test_dir, "n_photos": n_test},
        test_label_args=base_label_args + {"img_dir": test_dir, "save_to": test_labels},
    )

    return _fit_and_eval_model(
        data_module,
        resnet_variant,
        site_name,
        len(classes),
        learning_rate,
        weight_decay,
    )


def train_model_with_existing_data(
    resnet_variant: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ],
    learning_rate: float,
    weight_decay: float,
    site_name: str,
    train_dir: str | Path,
    test_dir: str | Path,
    classes: list[str],
) -> PhenoCamResNet:
    """Pipeline for building model with already downloaded/labeled data.

    :param resnet_variant: The ResNet variant to use as the model backbone.
    :type resnet_variant: str
    :param learning_rate: The learning rate to use.
    :type learning_rate: float
    :param weight_decay: The weight decay to use.
    :type weight_decay: float
    :param site_name: The name of the PhenoCam site you want.
    :type site_name: str
    :param train_dir: The path to the training images directory.
    :type train_dir: str | Path
    :param test_dir: The path to the test images directory.
    :type test_dir: str | Path
    :param classes: The image classes.
    :type classes: list[str]

    :return: The best model obtained during training.
    :rtype: PhenoCamResNet
    """
    if not isinstance(train_dir, Path):
        train_dir = Path(train_dir)
    if not isinstance(test_dir, Path):
        test_dir = Path(test_dir)
    data_module = PhenoCamDataModule(
        site_name,
        train_dir,
        train_dir / "labels.csv",
        test_dir,
        test_dir / "labels.csv",
    )
    data_module.prepare_data()

    return _fit_and_eval_model(
        data_module,
        resnet_variant,
        site_name,
        len(classes),
        learning_rate,
        weight_decay,
    )


def _fit_and_eval_model(
    data_module: PhenoCamDataModule,
    resnet_variant: Literal[
        "resnet18", "resnet34", "resnet52", "resnet101", "resnet152"
    ],
    site_name: str,
    n_classes: int,
    learning_rate: float,
    weight_decay: float,
) -> PhenoCamResNet:
    """Helper function for above public methods. Returns the best model."""
    data_module.setup(stage="fit")
    model = PhenoCamResNet(resnet_variant, n_classes, learning_rate, weight_decay)
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"{site_name}_lightning_logs")
    callbacks = [EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-8)]
    accelerator = "gpu" if torch.cuda.is_available() else None
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=50,
        log_every_n_steps=3,
        accelerator=accelerator,
        precision=16,
    )
    trainer.fit(model, data_module)

    data_module.setup(stage="test")
    trainer.test(datamodule=data_module, ckpt_path="best")

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_models_csv = Path("best_model_paths.csv")
    if not best_models_csv.exists():
        with open(best_models_csv, "w") as f:
            f.write("timestamp,site_name,best_model\n")
    with open(best_models_csv, "a") as f:
        f.write(f"{datetime.now().isoformat()},{site_name},{best_model_path}\n")

    best_model = PhenoCamResNet.load_from_checkpoint(best_model_path)
    best_model.freeze()
    print(f"Path of best model: {best_model_path}")
    return best_model


if __name__ == "__main__":
    main()
