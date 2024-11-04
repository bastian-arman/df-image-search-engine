import sys
import asyncio
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

# async def main():
#     conn = await db_connection(collection_name="delete_me", db_name="Test")
#     async for document in conn.find({"image_path": {"$exists": True}}):
#         print(document)

# if __name__ == "__main__":
#     asyncio.run(main())