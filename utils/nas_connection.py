import sys
import asyncio
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from PIL import Image
from smb.SMBConnection import SMBConnection
from io import BytesIO
from src.secret import NAS_IP, NAS_USERNAME, NAS_PASSWORD
from utils.logger import logging
from torch import Tensor
from PIL import ImageOps
from utils.validator import _check_gpu_memory
from sentence_transformers import SentenceTransformer, util


device = _check_gpu_memory()
model: str = "clip-ViT-B-32"
model = SentenceTransformer(model_name_or_path=model, device=device)


def nas_connector(
    username: str = NAS_USERNAME, password: str = NAS_PASSWORD, nas_ip: str = NAS_IP
) -> SMBConnection | None:
    try:
        logging.info("Openning connetion to NAS.")
        conn = SMBConnection(
            username=username,
            password=password,
            my_name="",
            remote_name="",
            use_ntlm_v2=True,
        )
        conn.connect(nas_ip)
        logging.info("Connected into NAS server.")
    except Exception as e:
        logging.error(f"Failed to connect to NAS: {e}")
        return None
    return conn


async def _directory_finder(
    nas_connector: SMBConnection, service_name: str, root_dir: str
) -> list:
    dir_path = [root_dir]

    try:
        for root_dirs in dir_path:
            path_list = nas_connector.listPath(
                service_name=service_name, path=root_dirs
            )
            for path in path_list:
                if path.isDirectory and path.filename not in (".", ".."):
                    new_dir = f"{root_dirs}/{path.filename}"
                    dir_path.append(new_dir)
                    logging.info(f"Found directory: {new_dir}")
        logging.info(f"Total dir found: {len(dir_path)}")
    except Exception as e:
        logging.error(f"Error while recursively finding directories: {e}")
    return dir_path


async def _image_finder(
    nas_connector: SMBConnection, service_name: str, image_dirs: list
) -> list:
    image_paths = []

    try:
        for directory in image_dirs:
            filepaths = nas_connector.listPath(
                service_name=service_name, path=directory
            )
            for file in filepaths:
                if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = f"{directory}/{file.filename}"
                    image_paths.append(image_path)
                    logging.info(f"Found image: {image_path}")

        logging.info(f"Total image found: {len(image_paths)}")
    except Exception as e:
        logging.error(f"Error while finding images: {e}")
    return image_paths


async def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image by resizing, grayscaling, and normalizing."""
    image = ImageOps.fit(image, (224, 224))
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageOps.invert(image)
    image = ImageOps.mirror(image)
    return image


async def normalize_embeddings(embeddings: Tensor) -> list | None:
    try:
        if embeddings.device.type == "cuda":
            embeddings = embeddings.cpu()

        embedding_norms = np.linalg.norm(
            x=embeddings.detach().numpy(), axis=1, keepdims=True
        )

        normalized_embeddings = embeddings.detach().numpy() / embedding_norms
        normalized_embeddings[np.isnan(normalized_embeddings)] = 0
    except Exception as e:
        logging.error(
            f"[normalize_embeddings] Error while normalizing embedding dims: {e}"
        )
        return None
    return normalized_embeddings


async def search_data(
    query: str, encoded_data: Tensor, image_paths: list, k=10
) -> list | None:
    """
    Search for similar images given a text or image query.

    Parameters:
    - query: Text or image data for querying similar images.
    - encoded_data: The embeddings of all images in the database.
    - image_paths: The paths of all images in the database, corresponding to the embeddings.
    - k: Number of top similar images to return.

    Returns:
    - List of tuples with the image path and similarity score for the top `k` similar images.
    """
    # Encode the query and normalize
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=True)
    print(type(query_emb))
    query_emb = await normalize_embeddings(query_emb)

    # Ensure encoded_data is normalized
    encoded_data = await normalize_embeddings(encoded_data)

    # Perform semantic search for top `k` similar images
    hits = util.semantic_search(query_emb, encoded_data, top_k=k)[0]

    # Map the top `k` hits to their image paths and similarity scores
    similar_images = [(image_paths[hit["corpus_id"]], hit["score"]) for hit in hits]

    return similar_images


async def _retrieve_data(
    nas_connector: SMBConnection, service_name: str, image_paths: list
) -> list | None:
    preprocessed_data = []

    try:
        for path in image_paths:
            file_obj = BytesIO()
            nas_connector.retrieveFile(
                service_name=service_name, path=path, file_obj=file_obj
            )
            file_obj.seek(0)
            image = Image.open(file_obj)
            preprocessed_data.append(await preprocess_image(image=image))

    except Exception as e:
        logging.error(
            f"[_retrieve_data] Error occured while retrieving data from NAS: {e}"
        )
        return None
    return preprocessed_data


async def main():
    service_name = "Dfactory"
    root_dir = "test_bastian"
    conn = await nas_connector()

    try:
        list_dir = await _directory_finder(
            nas_connector=conn, service_name=service_name, root_dir=root_dir
        )
        list_image = await _image_finder(
            nas_connector=conn, service_name=service_name, image_dirs=list_dir
        )

        preprocessed_images = await _retrieve_data(
            nas_connector=conn, service_name=service_name, image_paths=list_image
        )
        # processed_images = []
        # for image_paths in list_image:
        #     file_obj = BytesIO()
        #     conn.retrieveFile(
        #         service_name=service_name, path=image_paths, file_obj=file_obj
        #     )
        #     file_obj.seek(0)
        #     # print(file_obj)
        #     image = Image.open(file_obj)

        #     processed_image = await preprocess_image(image)
        #     processed_images.append(processed_image)

        encoded_data = model.encode(
            preprocessed_images,
            batch_size=8,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        print(encoded_data[:2])
        print(type(encoded_data))
        query = "white statue and fountain"
        k = 5  # Number of similar images to retrieve
        similar_images = await search_data(query, encoded_data, list_image, k=k)

        # Output the results
        for img_path, score in similar_images:
            print(f"Image Path: {img_path}, Similarity Score: {score}")
    finally:
        logging.info("Closed SMB connection.")
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
