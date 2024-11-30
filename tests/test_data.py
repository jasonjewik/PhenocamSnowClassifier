import pytest
from torchvision import transforms

from phenocam_snow.data import PhenoCamImageDataset


def test_image_dataset_len(tmp_path):
    img_dir = tmp_path
    labels_file = tmp_path / "labels.csv"

    # Refer to utils.label_images_via_subdir
    labels_file_content = """
        # Site: canadaojp
        # Categories:
        # 0. snow
        # 1. no_snow
        # 2. too_dark
        filename,label,int_label
        0.jpg,snow,0
        1.jpg,no_snow,1
        2.jpg,too_dark,2
        3.jpg,too_dark,2
        4.jpg,no_snow,1
    """

    with open(labels_file, "w") as f:
        for line in labels_file_content.splitlines():
            f.write(line.strip() + "\n")

    dataset = PhenoCamImageDataset(img_dir, labels_file)
    assert len(dataset) == 5


@pytest.mark.parametrize(
    ["transform", "expected_shape"],
    [
        (None, (1, 128, 128)),  # shape of the image returned by get_test_image
        (transforms.CenterCrop((64, 64)), (1, 64, 64)),
    ],
)
def test_image_dataset_get(transform, expected_shape, get_test_image, tmp_path):
    img_dir = tmp_path
    labels_file = tmp_path / "labels.csv"

    get_test_image.save(img_dir / "0.jpg")
    get_test_image.save(img_dir / "1.jpg")
    get_test_image.save(img_dir / "2.jpg")

    # Refer to utils.label_images_via_subdir
    labels_file_content = """
        # Site: canadaojp
        # Categories:
        # 0. snow
        # 1. no_snow
        # 2. too_dark
        filename,label,int_label
        0.jpg,no_snow,1
        2.jpg,snow,0
        1.jpg,too_dark,2
    """

    with open(labels_file, "w") as f:
        for line in labels_file_content.splitlines():
            f.write(line.strip() + "\n")

    dataset = PhenoCamImageDataset(img_dir, labels_file, transform)

    # Check labels
    assert dataset[0][1] == 1
    assert dataset[1][1] == 0
    assert dataset[2][1] == 2

    # Check images
    for img, _ in dataset:
        assert tuple(img.shape) == expected_shape
