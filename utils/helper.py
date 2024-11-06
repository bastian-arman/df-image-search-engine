import numpy as np
from utils.logger import logging
from PIL.Image import Image as PILImage
from PIL import ImageOps, Image
from smb.SMBConnection import SMBConnection
from sentence_transformers import SentenceTransformer, util
from torch import Tensor


async def _directory_finder(
    nas_connector: SMBConnection, service_name: str, root_dir: str
) -> list | None:
    """
    Recursively extract all directory to find dir path.

    Parameters:
    - nas_connector: NAS connection into streamlit server.
    - service_name: First directory of NAS. (e.g: /Dfactory/test_bastian/...) 'Dfactory' is the service name.
    - root_dir: Second directory of NAS. (e.g: /Dfactory/test_bastian/...) 'test_bastian' is the root dir.

    Returns: List of all directory available inside root_dir
    """
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
        logging.error(
            f"[_directory_finder] Error while recursively finding directories: {e}"
        )
        return None
    return dir_path


async def _image_finder(
    nas_connector: SMBConnection, service_name: str, image_dirs: list
) -> list | None:
    """
    Recuresively extract list of all image data.

    Parameters:
    - nas_connector: NAS connection into streamlit server.
    - service_name: First directory of NAS. (e.g: /Dfactory/test_bastian/...) 'Dfactory' is the service name.
    - image_dirs: Last directory for finding an image.
            (e.g:   /Dfactory/test_bastian/,
                    /Dfactory/test_bastian/2019
                    /Dfactory/test_bastian/2020
                    /Dfactory/test_bastian/2020/a
                    /Dfactory/test_bastian/2020/b
                            .   .   .   .
                    /Dfactory/test_bastian/2020/xxx)

    Returns: List of all image inside all given directory.
    """
    image_paths = []

    try:
        for directory in image_dirs:
            filepaths = nas_connector.listPath(
                service_name=service_name, path=directory
            )
            for file in filepaths:
                if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = f"{service_name}/{directory}/{file.filename}"
                    image_paths.append(image_path)
                    logging.info(f"Found image: {image_path}")

        logging.info(f"Total image found: {len(image_paths)}")
    except Exception as e:
        logging.error(f"[_image_finder] Error while finding images: {e}")
        return None
    return image_paths


async def _preprocess_image(image: Image) -> PILImage | None:
    """
    Preprocess image by resizing, grayscaling, and normalizing.

    Parameters:
    - image: Converted image into numpy type.

    Returns:
    - Preprocessed image using ImageOps.
    """
    try:
        image = ImageOps.fit(image, (224, 224))
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        image = ImageOps.invert(image)
        image = ImageOps.mirror(image)
    except Exception as e:
        logging.error(f"[_preprocess_image] Error while preprocessing an image: {e}")
        return None
    return image


async def _normalize_embeddings(embeddings: Tensor) -> list | None:
    """
    Normalize embedding dimension into 0 as min value and 1 as max value.

    Parameters:
    - embeddings: Tensor data from encoded images.

    Returns:
    - Normalized embedding data.
    """

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
            f"[_normalize_embeddings] Error while normalizing embedding dims: {e}"
        )
        return None
    return normalized_embeddings


async def _search_data(
    model: SentenceTransformer,
    query: str,
    encoded_data: Tensor,
    image_paths: list,
    return_data=10,
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
    try:
        query_emb = model.encode(
            sentences=[query], convert_to_tensor=True, show_progress_bar=True
        )
        query_emb = await _normalize_embeddings(query_emb)
        encoded_data = await _normalize_embeddings(encoded_data)
        hits = util.semantic_search(query_emb, encoded_data, top_k=return_data)[0]
        similar_images = [(image_paths[hit["corpus_id"]], hit["score"]) for hit in hits]
    except Exception as e:
        logging.error(f"[_search_data] Error while searching data: {e}")
        return None
    return similar_images
