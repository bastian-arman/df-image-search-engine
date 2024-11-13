import pytest
from random import randint
from utils.helper import _auto_update_encoding
from unittest.mock import patch

current_num = randint(90000, 100000)
encoded_list = ["encoded_data_PREVIEW_IMAGE_2927.npy"]


@pytest.mark.asyncio
async def test_trigger_a_valid_auto_encoding_if_any_available_cache_data() -> None:
    """Should return True due to differentiate data of current num (latest updated data on NAS) and initial num (data on NAS) is more than 10 data."""
    with patch("os.listdir", return_value=encoded_list):
        result = await _auto_update_encoding(
            cache_name=encoded_list, total_data_from_nas=current_num
        )
        assert result is True


@pytest.mark.asyncio
async def test_only_load_current_available_encoder() -> None:
    """Should return False due to available encoder data and the difference in files is 10 or fewer."""
    with patch("os.listdir", return_value=encoded_list):
        initial_num = int(encoded_list[0].split(".")[0].split("_")[-1])
        current_num = randint(initial_num, initial_num + 10)
        result = await _auto_update_encoding(
            cache_name=encoded_list, total_data_from_nas=current_num
        )
        assert result is False


@pytest.mark.asyncio
async def test_trigger_a_first_encoding() -> None:
    """Should return False due to no available saved encoder data."""
    encoded_list = []
    result = await _auto_update_encoding(
        cache_name=encoded_list, total_data_from_nas=current_num
    )
    assert result is False


@pytest.mark.asyncio
async def test_auto_update_encoding_with_exception() -> None:
    """Should return None due to an exception raised during execution."""
    with patch(
        "utils.helper._auto_update_encoding", side_effect=Exception("Mocked exception")
    ):
        result = await _auto_update_encoding(
            cache_name=["invalid_cache_name"], total_data_from_nas=current_num
        )
        assert result is None


@pytest.mark.asyncio
async def test_invalid_trigger_nas_data_lte_encoded_data() -> None:
    """Should return False due to current encoded data has more data than given data from NAS."""
    with patch("os.listdir", return_value=encoded_list):
        initial_num = int(encoded_list[0].split(".")[0].split("_")[-1])
        current_num = randint(initial_num - 100, initial_num - 50)
        result = await _auto_update_encoding(
            cache_name=encoded_list, total_data_from_nas=current_num
        )
        assert result is False
