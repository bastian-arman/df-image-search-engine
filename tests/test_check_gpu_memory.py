import pytest
from unittest.mock import patch
from utils.validator import _check_gpu_memory


@pytest.mark.asyncio
async def test_check_cuda_memory_with_cuda_cores_available() -> None:
    """Should return a float representing the usage of memory percentage."""
    with patch("torch.cuda.get_device_properties") as mock_get_device_properties, patch(
        "torch.cuda.memory_allocated"
    ) as mock_memory_allocated:
        mock_get_device_properties.return_value.total_memory = 100
        mock_memory_allocated.return_value = 40

        memory = await _check_gpu_memory(is_cuda_available=True)
        assert type(memory) is float
        assert memory == 0.4


@pytest.mark.asyncio
async def test_check_cuda_memory_with_cuda_cores_not_available() -> None:
    """Should return None because CUDA cores are not available."""
    memory = await _check_gpu_memory(is_cuda_available=False)
    assert memory is None


@pytest.mark.asyncio
async def test_check_cuda_memory_with_false_parameter_input() -> None:
    """Should return None due to invalid parameter type."""
    memory = await _check_gpu_memory(is_cuda_available=0.78)
    assert memory is None


@pytest.mark.asyncio
async def test_check_gpu_memory_raises_exception() -> None:
    """Should return None due to an exception raised inside _check_gpu_memory."""
    with patch(
        "torch.cuda.get_device_properties", side_effect=Exception("Mocked exception")
    ):
        memory = await _check_gpu_memory(is_cuda_available=False)
        assert memory is None
