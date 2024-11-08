import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.helper import _grab_all_images, _random_word


def test_grab_image_with_valid_image_directory() -> None:
    """
    Valid directory with image files.

    Should successfully grab a list of image path data.
    """

    image_paths = _grab_all_images(root_path="images")
    assert type(image_paths) is list


def test_grab_image_with_valid_directory_no_image_data() -> None:
    """
    Valid directory with no image files.

    Should return None due to image not found.
    """
    image_paths = _grab_all_images(root_path="scripts")
    assert image_paths is None


def test_grab_image_with_not_valid_directory() -> None:
    """
    Invalid directory path (e.g., random non-existent directory).

    Should return None due to inputted dir is not available.
    """
    random_dir_name = _random_word()
    image_paths = _grab_all_images(root_path=random_dir_name)
    assert image_paths is None


def test_grab_images_with_mixed_files() -> None:
    """
    Directory with mixed file extention.

    Should return only image paths (with extention jpeg, jpg, png).
    """
    image_paths = _grab_all_images(root_path="images")
    assert len(image_paths) == 1
