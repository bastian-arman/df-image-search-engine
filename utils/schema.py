from pydantic import BaseModel


class QueueData(BaseModel):
    query_embedding: list
    total_retrieved_data: int
