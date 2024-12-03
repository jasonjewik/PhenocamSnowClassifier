from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torchvision.io import decode_image

from phenocam_snow.data import PhenoCamDataModule
from phenocam_snow.model import PhenoCamResNet


def main():
    parser = ArgumentParser(description="Predicts image category using the given model")
    parser.add_argument(
        "site_name", help="The PhenoCam site for which we are generating predictions."
    )
    parser.add_argument("model_path", help="The path of the model to use.")
    parser.add_argument("--categories", nargs="+", help="The image categories to use.")
    parser.add_argument(
        "--directory",
        default=None,
        help="Provide this if you want to get predictions for all images in a local directory.",
    )
    parser.add_argument(
        "--urls", default=None, help="A file containing URLs, one per line."
    )
    args = parser.parse_args()

    model = load_model_from_file(args.model_path)
    if args.urls:
        run_model_online(model, args.site_name, args.categories, args.urls)
    elif args.directory:
        run_model_offline(model, args.site_name, args.categories, args.directory)


def classify_online(
    data_module: PhenoCamDataModule,
    model: PhenoCamResNet,
    categories: list[str],
    img_url: str,
) -> tuple[np.array, int]:
    """Performs online classification.

    :param data_module: A data module, used for pre-processing the image.
    :type data_module: PhenoCamDataModule
    :param model: The model to use.
    :type model: PhenoCamResNet
    :param categories: The categories to use.
    :type categories: list[str]
    :param img_url: The URL of the image to run classification on.
    :type img_url: str

    :return: A 2-tuple where the first element is the image at `img_url` as a
        NumPy array and the second element is the predicted label.
    :rtype: tuple[np.array, int]

    :raise requests.exception.Timeout: if the GET request to the given URL times out
    :raise RuntimeError: if a non-200 response is received from the GET request to the given URL
    """
    resp = requests.get(img_url, timeout=3)
    if resp.status_code == 200:
        img = Image.open(BytesIO(resp.content))
        np_img = np.array(img).T
        x = torch.from_numpy(np_img)
        x = data_module.preprocess(x.unsqueeze(0))
        yhat = model(x.to(model.device))
        pred = categories[torch.argmax(yhat, dim=1)]
        return (np_img, pred)
    else:
        raise RuntimeError(f"request to {img_url} got {resp.status_code} response")


def classify_offline(
    data_module: PhenoCamDataModule,
    model: PhenoCamResNet,
    categories: list[str],
    img_path: str | Path,
) -> tuple[np.array, int]:
    """Performs offline classification.

    :param data_module: A data module, used for pre-processing the image.
    :type data_module: PhenoCamDataModule
    :param model: The model to use.
    :type model: PhenoCamResNet
    :param categories: The image categories.
    :type categories: list[str]
    :param img_path: The file path of the image to classify.
    :type img_path: str | Path

    :return: A 2-tuple where the first element is the image at `img_path` as a
        NumPy array and the second element is the predicted label.
    :rtype: tuple[np.array, int]
    """
    x = data_module.preprocess(decode_image(img_path).unsqueeze(0))
    yhat = model(x.to(model.device))
    pred = categories[torch.argmax(yhat, dim=1)]
    return pred


def load_model_from_file(model_path: str | Path):
    """Loads a model from checkpoint file.

    :param model_path: The path to the model checkpoint file.
    :type model_path: str | Path

    :return: The loaded model.
    :rtype: PhenoCamResNet
    """
    model = PhenoCamResNet.load_from_checkpoint(model_path)
    model.eval()
    model.freeze()
    return model


def run_model_offline(
    model: PhenoCamResNet, site_name: str, categories: list[str], img_dir: str | Path
) -> pd.DataFrame:
    """Gets predicted labels for all images in a directory, writing the results to a CSV and returning as a DataFrame.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param site_name: The name of the PhenoCam site.
    :type site_name: str
    :param categories: The categories as strings.
    :type categories: list[str]
    :param img_dir: The directory containing the images to classify.
    :type img_dir: str | Path

    :return: A pandas DataFrame with the columns "filename" (str) and "predicted_label" (str).
    :rtype: pd.DataFrame
    """
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)

    data_module = PhenoCamDataModule(
        "site_name", "train_dir", "train_labels", "test_dir", "test_labels"
    )
    image_paths = list(img_dir.glob("*.jpg"))
    predictions = [
        classify_offline(data_module, model, categories, path) for path in image_paths
    ]

    data = [(path, pred) for path, pred in zip(image_paths, predictions)]
    df = pd.DataFrame(data, columns=["filename", "predicted_label"])
    save_to = img_dir.joinpath("predictions.csv")
    with open(save_to, "w+") as f:
        f.write(f"# Site: {site_name}\n")
        f.write("# Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"# {i}. {cat}\n")
    df.to_csv(save_to, mode="a", lineterminator="\n", index=False)
    print(f"Results written to {save_to}")

    return df


def run_model_online(
    model: PhenoCamResNet, site_name: str, categories: list[str], urls: str | Path
) -> pd.DataFrame:
    """Gets predicted labels for images online, writing the results to a CSV and returning as a DataFrame.

    :param model: The model to use.
    :type model: PhenoCamResNet
    :param site_name: The name of the PhenoCam site.
    :type site_name: str
    :param urls: The name of a file containing all the URLs, one per line.
    :type urls: str | Path

    :return: A pandas DataFrame with the columns "url" (str) and "predicted_label" (str).
    :rtype: pd.DataFrame
    """
    with open(urls) as f:
        links = [line.strip() for line in f.readlines()]

    data_module = PhenoCamDataModule(
        "site_name", "train_dir", "train_labels", "test_dir", "test_labels"
    )
    predictions = [
        classify_online(data_module, model, categories, link)[1] for link in links
    ]

    data = [(link, prediction) for link, prediction in zip(links, predictions)]
    df = pd.DataFrame(data, columns=["url", "predicted_label"])
    save_to = "predictions.csv"
    with open(save_to, "w+") as f:
        f.write(f"# Site: {site_name}\n")
        f.write("# Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"# {i}. {cat}\n")
    df.to_csv(save_to, mode="a", lineterminator="\n", index=False)
    print(f"Results written to {save_to}")

    return df


if __name__ == "__main__":
    main()
