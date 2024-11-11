from pydantic import BaseModel


class TextData(BaseModel):
    text_prompt: str
    total_data: int


# class ImageData(BaseModel):
#     image_data: Image
#     total_data: int
