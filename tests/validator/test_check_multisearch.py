# import pytest
# from utils.validator import _check_multisearch


# @pytest.mark.asyncio
# async def test_check_user_input_with_both_methods():
#     """Should return True because both methods are used."""
#     result = await _check_multisearch(image_description="text", image_uploader="image")
#     assert result is True


# @pytest.mark.asyncio
# async def test_check_user_input_with_only_image_uploader():
#     """Should return False because only image uploader is used."""
#     result = await _check_multisearch(image_description=None, image_uploader="image")
#     assert result is False


# @pytest.mark.asyncio
# async def test_check_user_input_with_only_image_description():
#     """Should return False because only image description is used."""
#     result = await _check_multisearch(image_description="text", image_uploader=None)
#     assert result is False


# @pytest.mark.asyncio
# async def test_check_user_input_no_data():
#     """Should return True because no data is provided."""
#     result = await _check_multisearch(image_description=None, image_uploader=None)
#     assert result is True


# @pytest.mark.asyncio
# async def test_check_user_input_only_spaces_in_image_description():
#     """Should return True because only whitespace is in image description."""
#     result = await _check_multisearch(image_description="   ", image_uploader=None)
#     assert result is True


# @pytest.mark.asyncio
# async def test_check_user_input_spaces_in_description_and_image_upload():
#     """Should return True because only the image uploader is used with whitespace in description."""
#     result = await _check_multisearch(image_description="   ", image_uploader="image")
#     assert result is True
