from datetime import datetime
from pydantic import BaseModel


class ImageData(BaseModel):
    created_at: datetime
    filename: str
    image_path: str
    image_embedding: list
    image_tag: str
    updated_at: datetime | None = None

class TagId(BaseModel):
    created_at: datetime
    corpus: str
    updated_at: datetime | None = None