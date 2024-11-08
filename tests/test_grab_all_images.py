import sys
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.helper import _grab_all_images, _random_word


def test_grab_image_with_valid_image_directory() -> None:
    """
    Should successfully grab a list of image path data.
    """

    image_paths = _grab_all_images(root_path="images")
    assert type(image_paths) == list

def test_grab_image_with_valid_directory_no_image_data() -> None:
    """
    Should return None due to image not found.
    """
    image_paths = _grab_all_images(root_path="scripts")
    assert image_paths == None

def test_grab_image_with_not_valid_directory() -> None:
    """
    Should return None due to inputted dir is not available.
    """
    random_dir_name = _random_word()
    image_paths = _grab_all_images(root_path=random_dir_name)
    assert image_paths == None
