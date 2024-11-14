import pytest
import numpy as np
import torch
from numpy import ndarray
from utils.helper import _normalize_embeddings


@pytest.mark.asyncio
async def test_normalize_embeddings_with_valid_tensor_cpu() -> None:
    """Should return normalized numpy array for valid CPU tensor input."""
    tensor_embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalized = _normalize_embeddings(tensor_embedding)
    assert isinstance(normalized, ndarray)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


@pytest.mark.asyncio
async def test_normalize_embeddings_with_valid_tensor_cuda() -> None:
    """Should return normalized numpy array for valid CUDA tensor input."""
    if torch.cuda.is_available():
        tensor_embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
        normalized = _normalize_embeddings(tensor_embedding)
        assert isinstance(normalized, ndarray)
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


@pytest.mark.asyncio
async def test_normalize_embeddings_with_valid_ndarray() -> None:
    """Should return normalized numpy array for valid ndarray input."""
    ndarray_embedding = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = _normalize_embeddings(ndarray_embedding)
    assert isinstance(normalized, ndarray)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


@pytest.mark.asyncio
async def test_normalize_embeddings_with_nan_values() -> None:
    """Should replace NaN values with zero in the normalized array."""
    tensor_embedding = torch.tensor([[np.nan, 1.0], [2.0, np.nan]])
    normalized = _normalize_embeddings(tensor_embedding)
    assert isinstance(normalized, ndarray)
    assert np.isnan(normalized).sum() == 0


@pytest.mark.asyncio
async def test_normalize_embeddings_with_unsupported_type() -> None:
    """Should return None and log an error for unsupported data type input."""
    unsupported_embedding = [1.0, 2.0, 3.0]
    normalized = _normalize_embeddings(unsupported_embedding)
    assert normalized is None


@pytest.mark.asyncio
async def test_normalize_embeddings_with_empty_array() -> None:
    """Should handle empty array input and return an empty array."""
    empty_ndarray = np.array([]).reshape(0, 2)
    normalized = _normalize_embeddings(empty_ndarray)
    assert isinstance(normalized, ndarray)
    assert normalized.size == 0


@pytest.mark.asyncio
async def test_normalize_embeddings_with_zero_norm() -> None:
    """Should handle embedding with zero norm by setting NaNs to zero."""
    tensor_embedding = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    normalized = _normalize_embeddings(tensor_embedding)
    assert isinstance(normalized, ndarray)
    assert np.all(normalized == 0)
