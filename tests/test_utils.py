from dataclasses import dataclass
from io import BytesIO

import pytest
import requests
from PIL import Image

from phenocam_snow.utils import (
    download,
    download_from_log,
    get_site_dates,
    get_site_images,
    get_site_months,
    get_site_names,
)


@dataclass
class MockHttpResponse:
    status_code: int
    content: bytes | None


@pytest.fixture
def get_test_image_as_bytes():
    bytes_buffer = BytesIO()
    img = Image.open("tests/data/test_img.jpg")
    img.save(bytes_buffer, format=img.format)
    return bytes_buffer.getvalue()


@pytest.fixture
def make_mock_http_get():
    def _make_mock_http_get(
        raises_timeout: bool, status_code: int, content: bytes | None = None
    ):
        def mock_http_get(*args, **kwargs):
            if raises_timeout:
                raise requests.exceptions.Timeout()
            return MockHttpResponse(status_code, content)

        return mock_http_get

    return _make_mock_http_get


def test_successful_get_site_names():
    actual_site_names = get_site_names()
    with open("tests/data/site_names.csv", "r") as f:
        expected_site_names = f.readlines()[0].split(",")
    assert actual_site_names == expected_site_names


@pytest.mark.parametrize(
    ["error_type", "match"],
    [
        (requests.exceptions.Timeout, None),
        (RuntimeError, r"did not get 200 response from .+"),
    ],
)
def test_unsuccessful_get_site_names(
    error_type, match, make_mock_http_get, monkeypatch
):
    if error_type == requests.exceptions.Timeout:
        mock_http_get = make_mock_http_get(True, None)
    elif error_type == RuntimeError:
        mock_http_get = make_mock_http_get(False, 404)
    monkeypatch.setattr("requests.get", mock_http_get)
    with pytest.raises(error_type, match=match):
        get_site_names()


def test_successful_get_site_months():
    actual_site_months = get_site_months(site_name="canadaojp")
    with open("tests/data/site_months.csv", "r") as f:
        expected_site_months = f.readlines()[0].split(",")
    assert actual_site_months == expected_site_months


@pytest.mark.parametrize(
    ["error_type", "match"],
    [
        (requests.exceptions.Timeout, None),
        (RuntimeError, r"did not get 200 response from .+"),
    ],
)
def test_unsuccessful_get_site_months(
    error_type, match, make_mock_http_get, monkeypatch
):
    if error_type == requests.exceptions.Timeout:
        mock_http_get = make_mock_http_get(True, None)
    elif error_type == RuntimeError:
        mock_http_get = make_mock_http_get(False, 404)
    monkeypatch.setattr("requests.get", mock_http_get)
    with pytest.raises(error_type, match=match):
        get_site_months(site_name="doesn't matter")


def test_successful_get_site_dates():
    actual_site_dates = get_site_dates(site_name="canadaojp", year="2024", month="01")
    with open("tests/data/site_dates.csv", "r") as f:
        expected_site_dates = f.readlines()[0].split(",")
    assert actual_site_dates == expected_site_dates


@pytest.mark.parametrize(
    ["error_type", "match"],
    [
        (requests.exceptions.Timeout, None),
        (RuntimeError, r"did not get 200 response from .+"),
    ],
)
def test_unsuccessful_get_site_dates(
    error_type, match, make_mock_http_get, monkeypatch
):
    if error_type == requests.exceptions.Timeout:
        mock_http_get = make_mock_http_get(True, None)
    elif error_type == RuntimeError:
        mock_http_get = make_mock_http_get(False, 404)
    monkeypatch.setattr("requests.get", mock_http_get)
    with pytest.raises(error_type, match=match):
        get_site_dates(site_name="does_not_matter", year="1970", month="01")


def test_successful_get_site_images():
    actual_site_images = get_site_images(
        site_name="canadaojp", year="2024", month="01", date="01"
    )
    with open("tests/data/site_images.csv", "r") as f:
        expected_site_images = f.readlines()[0].split(",")
    assert actual_site_images == expected_site_images


@pytest.mark.parametrize(
    ["error_type", "match"],
    [
        (requests.exceptions.Timeout, None),
        (RuntimeError, r"did not get 200 response from .+"),
    ],
)
def test_unsuccessful_get_site_images(
    error_type, match, make_mock_http_get, monkeypatch
):
    if error_type == requests.exceptions.Timeout:
        mock_http_get = make_mock_http_get(True, None)
    elif error_type == RuntimeError:
        mock_http_get = make_mock_http_get(False, 404)
    monkeypatch.setattr("requests.get", mock_http_get)
    with pytest.raises(error_type, match=match):
        get_site_images(site_name="does_not_matter", year="1970", month="01", date="01")


def test_successful_download(
    tmp_path, make_mock_http_get, get_test_image_as_bytes, monkeypatch
):
    site_name = "canadaojp"
    save_to = tmp_path
    n_photos = 2
    log_filename = "test.log"

    def mock_get_site_months(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01"]

    def mock_get_site_dates(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01/01"]

    def mock_get_site_images(*args, **kwargs):
        return [
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_000010.jpg",
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_003013.jpg",
        ]

    monkeypatch.setattr("phenocam_snow.utils.get_site_months", mock_get_site_months)
    monkeypatch.setattr("phenocam_snow.utils.get_site_dates", mock_get_site_dates)
    monkeypatch.setattr("phenocam_snow.utils.get_site_images", mock_get_site_images)
    monkeypatch.setattr(
        "requests.get", make_mock_http_get(False, 200, get_test_image_as_bytes)
    )

    assert download(site_name, save_to, n_photos, log_filename) == True

    with open(save_to / log_filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    assert len(lines) == 4
    assert lines[0] == f"INFO:Retrieved {site_name}'s per-month URLs"
    assert set(lines[1:3]) == set(
        f"INFO:Retrieved {url}" for url in mock_get_site_images()
    )
    assert lines[3] == f"INFO:Finished downloading {n_photos} photos"


@pytest.mark.parametrize(["n_photos"], [(0,), (-1,), (-100,)])
def test_download_with_bad_n_photos_arg(n_photos, tmp_path):
    site_name = "canadaojp"
    save_to = tmp_path
    log_filename = "test.log"
    with pytest.raises(
        ValueError, match="if n_photos is provided, it must be a positive integer"
    ):
        download(site_name, save_to, n_photos, log_filename)


@pytest.mark.parametrize(
    ["function"],
    [
        ("phenocam_snow.utils.get_site_months",),
        ("phenocam_snow.utils.get_site_dates",),
        ("phenocam_snow.utils.get_site_images",),
    ],
)
def test_download_with_fatal_timeouts(function, tmp_path, monkeypatch):
    site_name = "canadaojp"
    save_to = tmp_path
    n_photos = 2
    log_filename = "test.log"

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout()

    def mock_get_site_months(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01"]

    def mock_get_site_dates(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01/01"]

    def mock_get_site_images(*args, **kwargs):
        return [
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_000010.jpg",
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_003013.jpg",
        ]

    monkeypatch.setattr("phenocam_snow.utils.get_site_months", mock_get_site_months)
    monkeypatch.setattr("phenocam_snow.utils.get_site_dates", mock_get_site_dates)
    monkeypatch.setattr("phenocam_snow.utils.get_site_images", mock_get_site_images)

    # Override the function that should result in fatal timeout error
    monkeypatch.setattr(function, raise_timeout)

    assert download(site_name, save_to, n_photos, log_filename) == False

    with open(save_to / log_filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    if function == "phenocam_snow.utils.get_site_months":
        assert len(lines) == 1
        assert lines[0] == "ERROR:Call to get_site_months failed with Timeout"
    elif function == "phenocam_snow.utils.get_site_dates":
        assert len(lines) == 2
        assert lines[0] == f"INFO:Retrieved {site_name}'s per-month URLs"
        assert lines[1] == "ERROR:Call to get_site_dates failed with Timeout"
    elif function == "phenocam_snow.utils.get_site_images":
        assert len(lines) == 2
        assert lines[0] == f"INFO:Retrieved {site_name}'s per-month URLs"
        assert lines[1] == f"ERROR:Call to get_site_images failed with Timeout"


def test_download_with_nonfatal_timeouts(tmp_path, monkeypatch, make_mock_http_get):
    site_name = "canadaojp"
    save_to = tmp_path
    n_photos = 2
    log_filename = "test.log"

    def mock_get_site_months(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01"]

    def mock_get_site_dates(*args, **kwargs):
        return ["https://phenocam.nau.edu/webcam/browse/canadaojp/2024/01/01"]

    def mock_get_site_images(*args, **kwargs):
        return [
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_000010.jpg",
            "https://phenocam.nau.edu/data/archive/canadaojp/2024/01/canadaojp_2024_01_01_003013.jpg",
        ]

    monkeypatch.setattr("phenocam_snow.utils.get_site_months", mock_get_site_months)
    monkeypatch.setattr("phenocam_snow.utils.get_site_dates", mock_get_site_dates)
    monkeypatch.setattr("phenocam_snow.utils.get_site_images", mock_get_site_images)
    monkeypatch.setattr("requests.get", make_mock_http_get(True, None))

    assert download(site_name, save_to, n_photos, log_filename) == True

    with open(save_to / log_filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    assert len(lines) == 4
    assert lines[0] == f"INFO:Retrieved {site_name}'s per-month URLs"
    assert set(lines[1:3]) == set(
        f"WARN:request to {url} timed out" for url in mock_get_site_images()
    )
    assert lines[3] == f"WARN:Downloaded only 0 out of {n_photos} requested photos"


@pytest.mark.parametrize(
    ["timeout", "status_code", "has_content"],
    [(False, 200, True), (True, None, False), (False, 404, False)],
)
def test_download_from_log(
    timeout,
    status_code,
    has_content,
    tmp_path,
    monkeypatch,
    make_mock_http_get,
    get_test_image_as_bytes,
):
    save_to = tmp_path
    source_log = tmp_path / "source.log"
    log_filename = "out.log"
    images = ["img1.jpg", "img2.jpg"]

    with open(source_log, "w") as f:
        for img in images:
            f.write(f"INFO:Retrieved {img}\n")

    content = get_test_image_as_bytes if has_content else None
    monkeypatch.setattr(
        "requests.get", make_mock_http_get(timeout, status_code, content)
    )

    download_from_log(source_log, save_to, log_filename)

    with open(save_to / log_filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    assert len(lines) == 4
    assert lines[0] == f"INFO:Read {len(images)} image URLs from {str(source_log)}"
    if not timeout and status_code == 200 and has_content:
        assert set(lines[1:3]) == set(f"INFO:Retrieved {url}" for url in images)
        assert lines[3] == f"INFO:Finished downloading {len(images)} photos"
    elif timeout:
        assert set(lines[1:3]) == set(
            f"WARN:request to {url} timed out" for url in images
        )
        assert lines[3] == f"WARN:Downloaded only 0 out of {len(images)} URLs read"
    elif status_code == 404:
        assert set(lines[1:3]) == set(
            f"WARN:request to {url} got {status_code} response" for url in images
        )
        assert lines[3] == f"WARN:Downloaded only 0 out of {len(images)} URLs read"
    else:
        raise ValueError("unrecognized test case")
