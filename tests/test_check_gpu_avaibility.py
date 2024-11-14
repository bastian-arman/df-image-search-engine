import pytest
from unittest.mock import patch
from utils.validator import _check_gpu_availability


@pytest.mark.asyncio
async def test_gpu_available_with_memory() -> None:
    """Should return True if CUDA and memory are available."""
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=1
    ), patch(
        "torch.cuda.get_device_properties",
        return_value=type("DeviceProps", (), {"total_memory": 1024}),
    ):
        result = await _check_gpu_availability()
        assert result is True


@pytest.mark.asyncio
async def test_gpu_available_no_devices() -> None:
    """Should return False due to CUDA being available but no devices detected."""
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=0
    ):
        result = await _check_gpu_availability()
        assert result is False


@pytest.mark.asyncio
async def test_gpu_available_no_memory() -> None:
    """Should return False due to GPU device found but with no memory."""
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=1
    ), patch(
        "torch.cuda.get_device_properties",
        return_value=type("DeviceProps", (), {"total_memory": 0}),
    ):
        result = await _check_gpu_availability()
        assert result is False


@pytest.mark.asyncio
async def test_gpu_not_available() -> None:
    """Should return False because no CUDA cores are available."""
    with patch("torch.cuda.is_available", return_value=False):
        result = await _check_gpu_availability()
        assert result is False


@pytest.mark.asyncio
async def test_gpu_check_exception() -> None:
    """Should return None due to an exception being raised."""
    with patch("torch.cuda.is_available", side_effect=Exception("Mocked exception")):
        result = await _check_gpu_availability()
        assert result is None
