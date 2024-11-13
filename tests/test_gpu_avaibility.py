import pytest
from unittest.mock import patch
from utils.validator import _check_gpu_avaibility


@pytest.mark.asyncio
async def test_check_gpu_avaibility_with_valid_available_cuda_in_device() -> None:
    """Should return True if GPU is available, False if not."""
    with patch("torch.cuda.is_available", return_value=True):
        result = await _check_gpu_avaibility()
        assert result is True

    with patch("torch.cuda.is_available", return_value=False):
        result = await _check_gpu_avaibility()
        assert result is False


@pytest.mark.asyncio
async def test_check_gpu_avaibility_with_exception() -> None:
    """Should return None due to an exception being raised."""
    with patch("torch.cuda.is_available", side_effect=Exception("Mocked exception")):
        result = await _check_gpu_avaibility()
        assert result is None
