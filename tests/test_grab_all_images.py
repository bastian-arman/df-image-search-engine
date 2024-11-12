import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.helper import _grab_all_images, _random_word


@pytest.mark.asyncio
async def test_grab_image_with_valid_image_directory() -> None:
    """
    Valid directory with image files.

    Should successfully grab a list of image path data.
    """

    image_paths = await _grab_all_images(root_dir="images")
    assert type(image_paths) is list


@pytest.mark.asyncio
async def test_grab_image_with_valid_directory_no_image_data() -> None:
    """
    Valid directory with no image files.

    Should return None due to image not found.
    """
    image_paths = await _grab_all_images(root_dir="scripts")
    assert image_paths is None


@pytest.mark.asyncio
async def test_grab_image_with_not_valid_directory() -> None:
    """
    Invalid directory path (e.g., random non-existent directory).

    Should return None due to inputted dir is not available.
    """
    random_dir_name = _random_word()
    image_paths = await _grab_all_images(root_dir=random_dir_name)
    assert image_paths is None


@pytest.mark.asyncio
async def test_grab_images_with_mixed_files() -> None:
    """
    Directory with mixed file extention.

    Should return only image paths (with extention jpeg, jpg, png).
    """
    image_paths = await _grab_all_images(root_dir="images")
    assert len(image_paths) == 1
