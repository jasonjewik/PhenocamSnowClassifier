import os
import random
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def get_site_names() -> list[str]:
    """Gets all available PhenoCam site names.

    :param request_timeout_seconds: time to wait until the request is considered timed out, defaults to 3 seconds
    :type request_timeout_seconds: int

    :return: The list of PhenoCam site names
    :rtype: list[str]

    :raises requests.exceptions.Timeout: if the GET request times out
    :raises RuntimeError: if non-200 response is received
    """
    url = "https://phenocam.nau.edu/webcam/network/table/"
    pattern = re.compile(r'"\/webcam\/sites\/(.+)/"')
    resp = requests.get(url, timeout=3)
    if resp.status_code != 200:
        raise RuntimeError(f"did not get 200 response from {url}")
    return pattern.findall(resp.text)


def get_site_months(site_name: str) -> list[str]:
    """Gets all available months for a PhenoCam site.

    :param site_name: The name of the site to check.
    :type site_name: str

    :return: the list of URLs for the given site's months
    :rtype: list[str]

    :raises requests.exceptions.Timeout: if the GET request times out
    :raises RuntimeError: if non-200 response is received
    """
    url = f"https://phenocam.nau.edu/webcam/browse/{site_name}"
    pattern = re.compile(rf'"\/webcam\/browse\/{site_name}(\/.+\/.+)"')
    resp = requests.get(url, timeout=3)
    if resp.status_code != 200:
        raise RuntimeError(f"did not get 200 response from {url}")
    return [f"{url}{path}" for path in pattern.findall(resp.text)]


def get_site_dates(site_name: str, year: str, month: str) -> list[str]:
    """Gets all available dates for a PhenoCam site in a given year and month.

    :param site_name: The name of the site to check.
    :type site_name: str
    :param year: The year to retrieve dates for as a string like "YYYY".
    :type year: str
    :param month: The month to retrieve dates for as a zero-leading string like "01" or "12".
    :type month: str

    :return: the list of URLs for the given site's dates in the given year and month
    :rtype: list[str]

    :raises requests.exceptions.Timeout: if the GET request times out
    :raises RuntimeError: if non-200 response is received
    """
    url = f"https://phenocam.nau.edu/webcam/browse/{site_name}/{year}/{month}"
    pattern = re.compile(rf'"\/webcam\/browse\/{site_name}\/.+\/.+(\/.+)\/"')
    resp = requests.get(url, timeout=3)
    if resp.status_code != 200:
        raise RuntimeError(f"did not get 200 response from {url}")
    return [f"{url}{path}" for path in pattern.findall(resp.text)]


def get_site_images(site_name: str, year: str, month: str, date: str) -> list[str]:
    """Gets all available images for a PhenoCam site in a given year, month, and date.

    :param site_name: The name of the site to check.
    :type site_name: str
    :param year: The year to retrieve dates for as a string like "YYYY".
    :type year: str
    :param month: The month to retrieve dates for as a zero-leading string like "01" or "12".
    :type month: str
    :param date: The date to retrieve dates for as a zero-leading string like "01" or "30".
    :type date: str

    :return: the list of URLs for the given site's dates in the given year and month
    :rtype: list[str]

    :raises requests.exceptions.Timeout: if the GET request times out
    :raises RuntimeError: if non-200 response is received
    """
    base = "https://phenocam.nau.edu"
    url = f"{base}/webcam/browse/{site_name}/{year}/{month}/{date}"
    pattern = re.compile(rf'"(\/data\/archive\/{site_name}\/{year}\/{month}\/.*\.jpg)"')
    resp = requests.get(url, timeout=3)
    if resp.status_code != 200:
        raise RuntimeError(f"did not get 200 response from {url}")
    return [f"{base}{path}" for path in pattern.findall(resp.text)]


def download(
    site_name: str,
    save_to: str | Path,
    n_photos: int | None = None,
    log_filename: str | Path | None = None,
) -> bool:
    """Downloads photos taken in some time range at a given site.

    :param site_name: The name of the site to download from.
    :type site_name: str
    :param save_to: The destination directory for downloaded images. If the directory already exists, it is NOT
        cleared. New photos are added to the directory, except for duplicates, which are skipped.
    :type save_to: str | Path
    :param n_photos: The max number of photos to download, if specified. Otherwise, tries to retrieve all photos.
    :type n_photos: int | None
    :param log_filename: Logs will be emitted to this path, if specified, otherwise logs will be emitted to a file with a
        name like 'YYYY-mm-DDTHH-MM-SS.log' in the save_to destination directory.
    :type log_filename: str | Path | None

    :return: True if the download succeeded, False otherwise (see log file)
    :rtype: bool

    :raises ValueError: if n_photos is provided as a non-positive integer
    """
    if n_photos is not None and n_photos <= 0:
        raise ValueError("if n_photos is provided, it must be a positive integer")

    if type(save_to) is not Path:
        save_dir = Path(save_to)
    else:
        save_dir = save_to
    os.makedirs(save_dir, exist_ok=True)

    log_filename = (
        log_filename
        or f'{datetime.now().isoformat().split(".")[0].replace(":", "-")}.log'
    )
    log_filepath = save_dir.joinpath(log_filename)

    log_file = open(log_filepath, "a")
    write_info = lambda msg: log_file.write(f"INFO:{msg}\n")
    write_warn = lambda msg: log_file.write(f"WARN:{msg}\n")
    write_error = lambda msg: log_file.write(f"ERROR:{msg}\n")

    try:
        site_month_urls = get_site_months(site_name)
    except Exception as e:
        write_error(f"Call to get_site_months failed with {e.__class__.__name__}")
        return False

    write_info(f"Retrieved {site_name}'s per-month URLs")

    image_urls = []
    year_month_str = rf"https:\/\/phenocam.nau.edu\/webcam\/browse\/{site_name}\/(?P<year>.+)\/(?P<month>.+)"
    year_month_pattern = re.compile(year_month_str)
    date_pattern = re.compile(year_month_str + r"\/(?P<date>.+)")

    # TODO:implement multiprocessing to hasten downloads
    for month_url in tqdm(site_month_urls, desc="months"):
        match = year_month_pattern.match(month_url)
        year, month = match.group("year"), match.group("month")
        try:
            site_date_urls = get_site_dates(site_name, year, month)
        except Exception as e:
            write_error(f"Call to get_site_dates failed with {e.__class__.__name__}")
            return False
        for date_url in tqdm(site_date_urls, desc="dates", leave=False):
            match = date_pattern.match(date_url)
            date = match.group("date")
            try:
                image_urls.extend(get_site_images(site_name, year, month, date))
            except Exception as e:
                write_error(
                    f"Call to get_site_images failed with {e.__class__.__name__}"
                )
                return False

    random.shuffle(image_urls)
    n_downloaded = 0

    for url in image_urls:
        try:
            resp = requests.get(url, timeout=3)
        except requests.exceptions.Timeout:
            write_warn(f"request to {url} timed out")
            continue
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.save(save_dir.joinpath(os.path.basename(url)))
            n_downloaded += 1
            write_info(f"Retrieved {url}")
        else:
            write_warn(f"request to {url} got {resp.status_code} response")
        if n_photos is not None and n_downloaded == n_photos:
            break

    if n_downloaded < n_photos:
        write_warn(f"Downloaded only {n_downloaded} out of {n_photos} requested photos")
    else:
        write_info(f"Finished downloading {n_downloaded} photos")

    log_file.close()

    return True


def download_from_log(
    source_log: str | Path, save_to: str | Path, log_filename: str | Path | None = None
) -> None:
    """Downloads images that are listed in a log file.

    :param source_log: The log file to get image URLs from.
    :type source_log: str
    :param save_to: The destination directory for downloaded images.
    :type save_to: str
    :param log_filename: Logs will be emitted to this path, if specified, otherwise logs will be emitted to a file with a
        name like 'YYYY-mm-DDTHH-MM-SS.log' in the save_to destination directory.
    :type log_filename: str | Path | None
    """
    if type(save_to) is not Path:
        save_dir = Path(save_to)
    else:
        save_dir = save_to
    if not save_dir.is_dir():
        os.mkdir(save_dir)

    log_filename = (
        log_filename
        or f'{datetime.now().isoformat().split(".")[0].replace(":", "-")}.log'
    )
    log_filepath = save_dir.joinpath(log_filename)

    img_urls = []
    with open(source_log, "r") as f:
        for line in f:
            if line.startswith("INFO:Retrieved "):
                url = line.split(" ")[1].strip()
                img_urls.append(url)

    log_file = open(log_filepath, "a")
    write_info = lambda msg: log_file.write(f"INFO:{msg}\n")
    write_warn = lambda msg: log_file.write(f"WARN:{msg}\n")

    write_info(f"Read {len(img_urls)} image URLs from {str(source_log)}")

    n_downloaded = 0
    for url in img_urls:
        try:
            resp = requests.get(url, timeout=3)
        except requests.exceptions.Timeout:
            write_warn(f"request to {url} timed out")
            continue
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.save(save_dir.joinpath(os.path.basename(url)))
            write_info(f"Retrieved {url}")
            n_downloaded += 1
        else:
            write_warn(f"request to {url} got {resp.status_code} response")

    if n_downloaded < len(img_urls):
        write_warn(f"Downloaded only {n_downloaded} out of {len(img_urls)} URLs read")
    else:
        write_info(f"Finished downloading {n_downloaded} photos")

    log_file.close()


def label_images(
    site_name: str,
    categories: list[str],
    img_dir: str | Path,
    save_to: str | Path,
    bypass_user_prompt: bool = False,
) -> None:
    """Allows the user to label images by moving them into the appropriate subdirectory. After the labels have been
    recorded into an annotations file, the images are moved back into the original directory.

    :param site_name: The name of the site.
    :type site_name: str
    :param categories: The image categories.
    :type categories: list[str]
    :param img_dir: The directory containing the image subdirectories.
    :type img_dir: str | Path
    :param save_to: The destination path for the labels file.
    :type save_to: str | Path
    :param bypass_user_prompt: whether to bypass the user prompt (for testing purposes, defaults to False)
    :type bypass_user_prompt: bool
    """
    if type(img_dir) is not Path:
        img_dir = Path(img_dir)
    assert img_dir.is_dir()

    dircats = []
    for cat in categories:
        dircat = img_dir.joinpath(Path(cat))
        dircats.append(dircat)
        if not dircat.exists() or not dircat.is_dir():
            os.mkdir(dircat)

    if not bypass_user_prompt:
        input(
            "Move images into the appropriate sub-directory then press any key to continue."
        )

    filenames = []
    for dircat in dircats:
        filenames.extend([os.path.basename(fpath) for fpath in dircat.glob("*.jpg")])
    df = pd.DataFrame(
        zip(filenames, categories), columns=["filename", "label"]
    ).explode("filename")
    with open(save_to, "w+") as f:
        f.write(f"# Site: {site_name}\n")
        f.write("# Categories:\n")
        for i, cat in enumerate(categories):
            f.write(f"# {i}. {cat}\n")
    df.to_csv(save_to, mode="a", index=False)

    for item in img_dir.glob("*"):
        if item.is_dir():
            for subitem in sorted(item.glob("*")):
                new_path = Path(subitem.resolve().parent.parent).joinpath(subitem.name)
                subitem.rename(new_path)


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
