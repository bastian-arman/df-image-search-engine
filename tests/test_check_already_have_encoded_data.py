import pytest
from unittest.mock import patch
from utils.validator import _check_already_have_encoded_data

encoded_list = [
    "encoded_data_PREVIEW_IMAGE_2927.npy",
    "encoded_data_PREVIEW_IMAGE_3100.npy",
    "encoded_data_PREVIEW_IMAGE_3200.npy",
    "encoded_data_testing_3200.npy",
    "encoded_data_testing_3300.npy",
    "encoded_data_testing_3400.npy",
]


@pytest.mark.asyncio
async def test_find_valid_similar_encoding_data_on_already_saved_encoder_data() -> None:
    """Should return a list of 3 similar encoded data items in the /cache directory."""
    with patch("os.listdir", return_value=encoded_list):
        root_dir = "PREVIEW_IMAGE"
        result = _check_already_have_encoded_data(
            root_dir=root_dir, encoded_list=encoded_list
        )
        assert len(result) == 3


@pytest.mark.asyncio
async def test_find_similar_encoding_data_with_invalid_root_directory() -> None:
    """Should return None since root_dir does not contain similar encoded data."""
    with patch("os.listdir", return_value=encoded_list):
        root_dir = "random_root_dir"
        result = _check_already_have_encoded_data(
            root_dir=root_dir, encoded_list=encoded_list
        )
        assert result is None


@pytest.mark.asyncio
async def test_find_similar_encoding_data_with_invalid_encoding_list() -> None:
    """Should return None since encoded_list takes empty list of data."""
    root_dir = "PREVIEW_IMAGE"
    encoded_list = []
    result = _check_already_have_encoded_data(
        root_dir=root_dir, encoded_list=encoded_list
    )
    assert result is None
