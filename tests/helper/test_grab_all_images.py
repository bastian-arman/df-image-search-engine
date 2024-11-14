import os
import pytest
from unittest.mock import patch
from utils.helper import _grab_all_images, _random_word


@pytest.mark.asyncio
async def test_grab_image_with_valid_image_directory() -> None:
    """Should successfully grab a list of image path data."""
    image_paths = await _grab_all_images(root_dir="images")
    assert type(image_paths) is list


@pytest.mark.asyncio
async def test_grab_image_with_valid_directory_no_image_data() -> None:
    """Should return None due to no images found in the directory."""
    image_paths = await _grab_all_images(
        root_dir="images/empty_dir_used_as_unit_testing"
    )
    assert image_paths is None


@pytest.mark.asyncio
async def test_grab_image_with_not_valid_directory() -> None:
    """Should return None because the directory does not exist."""
    random_dir_name = _random_word()
    image_paths = await _grab_all_images(root_dir=random_dir_name)
    assert image_paths is None


@pytest.mark.asyncio
async def test_grab_images_with_mixed_files() -> None:
    """Should return only paths to image files (jpeg, jpg, png) ignoring other files."""
    image_paths = await _grab_all_images(root_dir="images")
    total_image_files = [
        file
        for root, dirs, files in os.walk("images")
        for file in files
        if file.endswith(("jpeg", "jpg", "png"))
    ]
    assert len(image_paths) == len(total_image_files)


@pytest.mark.asyncio
async def test_grab_images_in_nested_directories() -> None:
    """Should successfully find images in nested directories."""
    image_paths = await _grab_all_images(root_dir="images")
    assert type(image_paths) is list


@pytest.mark.asyncio
async def test_grab_images_with_hidden_files_and_directories() -> None:
    """Should ignore hidden files and directories."""
    image_paths = await _grab_all_images(root_dir="images")
    assert not any(path in path for path in image_paths if path.startswith("."))


@pytest.mark.asyncio
async def test_grab_images_with_unsupported_files() -> None:
    """Should return only supported image files, ignoring unsupported files."""
    image_paths = await _grab_all_images(root_dir="images")
    assert not any(
        file for file in image_paths if file.endswith((".txt", ".bmp", ".py"))
    )


@pytest.mark.asyncio
async def test_grab_images_with_permission_denied() -> None:
    """Should handle permission errors gracefully and return None."""
    with patch("os.path.exists", return_value=True), patch(
        "os.listdir", side_effect=PermissionError("Permission denied")
    ):
        image_paths = await _grab_all_images(root_dir="restricted_dir")
        assert image_paths is None
