import pytest
from PIL import Image


@pytest.fixture(scope="session")
def get_test_image():
    return Image.open("tests/data/test_img.jpg")
