import pytest
from unittest.mock import patch
from utils.validator import _check_gpu_memory


@pytest.mark.asyncio
async def test_check_cuda_memory_with_valid_data() -> None:
    """Should return a float representing memory usage percentage."""
    with patch("torch.cuda.get_device_properties") as mock_get_device_properties, patch(
        "torch.cuda.memory_allocated"
    ) as mock_memory_allocated:
        mock_get_device_properties.return_value.total_memory = 100
        mock_memory_allocated.return_value = 40
        memory = await _check_gpu_memory(is_cuda_available=True)
        assert type(memory) is float
        assert memory == 0.4


@pytest.mark.asyncio
async def test_check_cuda_memory_with_zero_total_memory() -> None:
    """Should return None when total memory is zero, avoiding division by zero."""
    with patch("torch.cuda.get_device_properties") as mock_get_device_properties, patch(
        "torch.cuda.memory_allocated", return_value=40
    ):
        mock_get_device_properties.return_value.total_memory = 0
        memory = await _check_gpu_memory(is_cuda_available=True)
        assert memory is None


@pytest.mark.asyncio
async def test_check_cuda_memory_with_allocated_exceeding_total() -> None:
    """Should return None when allocated memory exceeds total memory."""
    with patch("torch.cuda.get_device_properties") as mock_get_device_properties, patch(
        "torch.cuda.memory_allocated"
    ) as mock_memory_allocated:
        mock_get_device_properties.return_value.total_memory = 50
        mock_memory_allocated.return_value = 60
        memory = await _check_gpu_memory(is_cuda_available=True)
        assert memory is None


@pytest.mark.asyncio
async def test_check_cuda_memory_with_large_memory_values() -> None:
    """Should handle large memory values without error."""
    with patch("torch.cuda.get_device_properties") as mock_get_device_properties, patch(
        "torch.cuda.memory_allocated"
    ) as mock_memory_allocated:
        mock_get_device_properties.return_value.total_memory = 1_000_000_000
        mock_memory_allocated.return_value = 500_000_000
        memory = await _check_gpu_memory(is_cuda_available=True)
        assert memory == 0.5


@pytest.mark.asyncio
async def test_check_cuda_memory_with_invalid_cuda_available_type() -> None:
    """Should return None due to invalid is_cuda_available type."""
    memory = await _check_gpu_memory(is_cuda_available="yes")
    assert memory is None


@pytest.mark.asyncio
async def test_check_cuda_memory_when_memory_allocated_raises_exception() -> None:
    """Should return None due to an exception in memory_allocated."""
    with patch(
        "torch.cuda.memory_allocated", side_effect=Exception("Mocked exception")
    ):
        memory = await _check_gpu_memory(is_cuda_available=True)
        assert memory is None
