def download(
    site_name: str,
    save_to: str | Path,
    n_photos: int | None = None,
    log_filename: str | Path | None = None,
) -> bool:
    """Downloads photos at a given site.

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

    total_processes = 8
    process_manager = multiprocessing.Manager()
    list_proxy: ListProxy[str] = process_manager.list()
    print(f"Process manager started {total_processes} processes.")

    processes = []
    for process_id in range(total_processes):
        months_for_this_process = site_month_urls[process_id::total_processes]
        process = multiprocessing.Process(
            target=_download_worker,
            args=(process_id, site_name, months_for_this_process, list_proxy),
        )
        processes.append(process)
        process.start()

    for current_process in processes:
        current_process.join()

    image_urls = []
    for item in list(list_proxy):
        if item.endswith("jpg"):
            image_urls.append(item)
        else:
            write_error(item)
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
    if not img_dir.is_dir():
        raise ValueError(f"img_dir {img_dir} is not a directory")

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

    data = []
    for dircat in dircats:
        for fpath in dircat.glob("*.jpg"):
            data.append((os.path.basename(fpath), os.path.basename(dircat)))
    df = pd.DataFrame(data, columns=["filename", "label"])
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
            item.rmdir()


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
