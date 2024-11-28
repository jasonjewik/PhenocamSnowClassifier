from dataclasses import dataclass

import pytest
import requests

from phenocam_snow.utils import (
    get_site_dates,
    get_site_images,
    get_site_months,
    get_site_names,
)


@dataclass
class MockHttpResponse:
    status_code: int


@pytest.fixture
def make_mock_http_get():
    def _make_mock_http_get(raises_timeout: bool, status_code: int):
        def mock_http_get(*args, **kwargs):
            if raises_timeout:
                raise requests.exceptions.Timeout()
            return MockHttpResponse(status_code)

        return mock_http_get

    return _make_mock_http_get


def test_successful_get_site_names():
    actual_site_names = get_site_names()
    with open("tests/site_names.csv", "r") as f:
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
    with open("tests/site_months.csv", "r") as f:
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
    with open("tests/site_dates.csv", "r") as f:
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
    with open("tests/site_images.csv", "r") as f:
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
