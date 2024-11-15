import pytest
from unittest.mock import patch
from utils.helper import _wrapper_queue_data


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_valid_data() -> None:
    """Should return a dictionary with query_embedding and total_retrieved_data because valid data is passed."""
    mock_embedding = [1.2819, 0.217891, 2.37198]
    mock_total_retrieved_data = 19

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert type(data) is dict
    assert data == {
        "query_embedding": mock_embedding,
        "total_retrieved_data": mock_total_retrieved_data,
    }


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_empty_embedding() -> None:
    """Should return None because the query_embedding is empty."""
    mock_embedding = []
    mock_total_retrieved_data = 19

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_none_embedding() -> None:
    """Should return None because the query_embedding is None."""
    mock_embedding = None
    mock_total_retrieved_data = 19

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_invalid_embedding_type() -> None:
    """Should return None because the query_embedding is not of type list."""
    mock_embedding = "invalid_embedding"
    mock_total_retrieved_data = 19

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_negative_total_retrieved_data() -> None:
    """Should return None because the total_retrieved_data is negative, which is invalid."""
    mock_embedding = [1.2819, 0.217891, 2.37198]
    mock_total_retrieved_data = -5

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_non_integer_total_retrieved_data() -> None:
    """Should return None because the total_retrieved_data is not an integer."""
    mock_embedding = [1.2819, 0.217891, 2.37198]
    mock_total_retrieved_data = "string_instead_of_integer"

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_large_data() -> None:
    """Should return a dictionary with query_embedding and total_retrieved_data because large data is handled successfully."""
    mock_embedding = [i for i in range(10000)]
    mock_total_retrieved_data = 99999

    data = await _wrapper_queue_data(
        query_embedding=mock_embedding, total_retrieved_data=mock_total_retrieved_data
    )
    assert type(data) is dict
    assert data["query_embedding"] == mock_embedding
    assert data["total_retrieved_data"] == mock_total_retrieved_data


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_missing_required_key() -> None:
    """Should return None because a required key is missing from the input (simulated by mocking the data)."""
    with patch(
        "utils.helper.QueueData.model_dump",
        side_effect=KeyError("Missing required field"),
    ):
        mock_embedding = [1.2819, 0.217891, 2.37198]
        mock_total_retrieved_data = 19

        data = await _wrapper_queue_data(
            query_embedding=mock_embedding,
            total_retrieved_data=mock_total_retrieved_data,
        )
        assert data is None


@pytest.mark.asyncio
async def test_wrapper_queue_data_with_exception_in_queue_data() -> None:
    """Should return None if an exception is raised while creating the QueueData instance."""
    with patch(
        "utils.helper.QueueData", side_effect=Exception("QueueData creation failed")
    ):
        mock_embedding = [1.2819, 0.217891, 2.37198]
        mock_total_retrieved_data = 19

        data = await _wrapper_queue_data(
            query_embedding=mock_embedding,
            total_retrieved_data=mock_total_retrieved_data,
        )
        assert data is None
