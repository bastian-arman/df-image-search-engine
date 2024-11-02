import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from motor.motor_asyncio import AsyncIOMotorClient
from src.secret import (
    MONGODB_DATABASE,
    MONGOODB_USERNAME, 
    MONGODB_PASSWORD
)

"""
Database connection into local mongodb.
"""

async def db_connection(
    collection_name: str,
    db_name: str = MONGODB_DATABASE,
    mongodb_username: str = MONGOODB_USERNAME,
    mongodb_password: str = MONGODB_PASSWORD
) -> AsyncIOMotorClient:
    try:
        client = AsyncIOMotorClient(f"mongodb://{mongodb_username}:{mongodb_password}@localhost:27017")
        db_client = client[db_name]
        collection = db_client[collection_name]
    except Exception as E:
        return f"An error occurred: {E}"
    return collection

# async def insert_dummy_data(collection: str):
#     dummy_data = {
#         "tag_id": "test_tag_001",
#         "description": "A test image description",
#         "image_path": "/path/to/test/image.jpg",
#         "embedding_data": [0.1, 0.2, 0.3, 0.4, 0.5]  # Example embedding vector
#     }

#     collection = await db_connection(collection_name=collection)
#     result = await collection.insert_one(dummy_data)

#     print(f"Inserted document with id: {result.inserted_id}")

# if __name__ == "__main__":
#     collection_name = "tag_id"  # This should be the name of the collection you want to insert into
#     asyncio.run(insert_dummy_data(collection=collection_name))
