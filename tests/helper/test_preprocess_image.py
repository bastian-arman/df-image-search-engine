from unittest.mock import patch
from PIL import Image
from io import BytesIO
from utils.helper import _preprocess_image

image_path = "images/Architecture Flow.png"


def test_preprocess_image_with_valid_image() -> None:
    """Should return Image.Image data because the image is processed correctly."""
    opened_image = Image.open(image_path)
    processed_image = _preprocess_image(opened_image)
    assert processed_image is not None
    assert isinstance(processed_image, Image.Image)


def test_preprocess_image_with_invalid_image() -> None:
    """Should return None because the image is invalid or corrupted."""
    invalid_image = BytesIO(b"not an image")
    processed_image = _preprocess_image(invalid_image)
    assert processed_image is None


def test_preprocess_image_with_none_input() -> None:
    """Should return None because the input is None."""
    processed_image = _preprocess_image(None)
    assert processed_image is None


def test_preprocess_image_with_unusual_aspect_ratio() -> None:
    """Should return Image.Image because the image size is changed to (224, 224) after preprocessing."""
    opened_image = Image.open(image_path)
    processed_image = _preprocess_image(opened_image)
    assert processed_image.size == (224, 224)


def test_preprocess_image_with_grayscale_conversion() -> None:
    """Should return Image.Image because the image is converted to grayscale during preprocessing."""
    opened_image = Image.open(image_path)
    processed_image = _preprocess_image(opened_image)
    assert processed_image.mode == "L"


def test_preprocess_image_with_flip_and_mirror() -> None:
    """Should return a processed image different from the original because flip and mirror transformations are applied."""
    opened_image = Image.open(image_path)
    processed_image = _preprocess_image(opened_image)
    assert processed_image != opened_image


def test_preprocess_image_with_invalid_type() -> None:
    """Should return None because the input type is invalid (not an image)."""
    invalid_input = 1234
    processed_image = _preprocess_image(invalid_input)
    assert processed_image is None


def test_preprocess_image_with_empty_image() -> None:
    """Should return None because the image is empty with size (0, 0)."""
    image = Image.new("RGB", (0, 0))
    processed_image = _preprocess_image(image)
    assert processed_image is None


def test_preprocess_image_with_unsupported_format() -> None:
    """Should return None because the image format is unsupported."""
    with patch("PIL.Image.open", side_effect=IOError("Unsupported format")):
        processed_image = _preprocess_image(None)
        assert processed_image is None
