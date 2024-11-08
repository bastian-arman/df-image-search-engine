import os
import sys
import random
import string
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger import logging
from PIL.Image import Image as PILImage
from PIL import ImageOps, Image
from numpy import ndarray
from numpy.linalg import norm
from numpy import isnan
from sentence_transformers import util
from torch import Tensor


def _setup_sidebar() -> None:
    """
    Create sidebar project description.
    """
    try:
        with st.sidebar:
            st.header("Image Similarity Search Engine")
            st.divider()
            st.subheader("Project Overview")
            st.write(
                """
                This project demonstrates a ***Proof of Concept (PoC)*** for an advanced Image Similarity Search Engine tailored for private datasets.
                Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images.
                """
            )
            st.divider()
    except Exception as e:
        logging.error(f"[_setup_sidebar] Error while setup sidebar: {e}")
    return None


def _grab_all_images(root_path: str) -> list | None:
    """
    Recursively extracting all image path based on root path dir.

    Parameters:
    - root_path: Root directory for searching image data.

    Returns:
    - List of all image data in extention jpg, jpeg, png.
    """
    try:
        if not os.path.exists(path=root_path):
            logging.error(
                f"[_grab_all_images] Directory {root_path} not available, make sure its available in projects directory."
            )
            return None

        image_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = [
            str(path)
            for path in Path(root_path).rglob("*")
            if path.suffix.lower() in image_extensions
        ]

        if not image_paths:
            logging.error(f"[_grab_all_images] No image files found in {root_path}")
            return None

    except Exception as e:
        logging.error(
            f"[_grab_all_images] Error occurred while grabbing all image data: {e}"
        )
        return None

    return image_paths


def _preprocess_image(image: Image) -> PILImage | None:
    """
    Preprocess image by resizing, grayscaling, and normalizing etc.

    Parameters:
    - image: Converted image into numpy type.

    Returns:
    - Preprocessed image using ImageOps.
    """
    try:
        image = ImageOps.fit(image, (224, 224))
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        # image = ImageOps.flip(image)
        image = ImageOps.expand(image)
        image = ImageOps.mirror(image)
    except Exception as e:
        logging.error(f"[_preprocess_image] Error while preprocessing an image: {e}")
        return None
    return image


def _normalize_embeddings(embeddings: Tensor | ndarray) -> ndarray | None:
    """
    Normalize embedding tensor from range 1 into -1

    Parameters:
    - embeddings: Tensor or
    """
    try:
        if not isinstance(embeddings, (Tensor, ndarray)):
            logging.error(
                f"Expected embeddings to be Tensor or ndarray, got {type(embeddings)}"
            )
            return None

        if isinstance(embeddings, Tensor) and embeddings.device.type == "cuda":
            embeddings = embeddings.cpu()

        embedding_norms = norm(
            x=embeddings.detach().numpy()
            if isinstance(embeddings, Tensor)
            else embeddings,
            axis=1,
            keepdims=True,
        )

        normalized_embeddings = (
            embeddings.detach().numpy() / embedding_norms
            if isinstance(embeddings, Tensor)
            else embeddings / embedding_norms
        )
        normalized_embeddings[isnan(normalized_embeddings)] = 0
    except Exception as e:
        logging.error(
            f"[_normalize_embeddings] Error while normalizing embedding dims: {e}"
        )
        return None
    return normalized_embeddings


async def _search_data(
    query_emb: Tensor,
    encoded_data: ndarray,
    image_paths: list,
    return_data=10,
) -> list | None:
    """
    Search for similar images based on a precomputed query embedding.

    Parameters:
    - query_emb: The embedding of the query (text or image).
    - encoded_data: The embeddings of all images in the database.
    - image_paths: The paths of all images in the database, corresponding to the embeddings.
    - return_data: Number of top similar images to return.

    Returns:
    - List of tuples with the image path and similarity score for the top `return_data` similar images.
    """
    try:
        query_emb = _normalize_embeddings(embeddings=query_emb)
        encoded_data = _normalize_embeddings(embeddings=encoded_data)
        hits = util.semantic_search(
            query_embeddings=query_emb,
            corpus_embeddings=encoded_data,
            top_k=return_data,
        )[0]
        similar_images = [(image_paths[hit["corpus_id"]], hit["score"]) for hit in hits]
    except Exception as e:
        logging.error(f"[_search_data] Error while searching data: {e}")
        return None
    return similar_images


def _random_word(length: int = 4) -> str:
    if length < 1:
        raise ValueError("length parameter should be more than 0")

    alphabet = string.ascii_lowercase
    word = "".join(random.choice(alphabet) for _ in range(length))

    return word
