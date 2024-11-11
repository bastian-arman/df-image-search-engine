import os
import sys
import random
import string
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from numpy import isnan
from torch import Tensor
from numpy import ndarray
from typing import Literal
from numpy.linalg import norm
from PIL import ImageOps, Image
from utils.logger import logging
from utils.schema import TextData
from src.secret import RABBITMQ_HOST
from sentence_transformers import util
from PIL.Image import Image as PILImage
from pika import BlockingConnection, ConnectionParameters, BasicProperties
from sentence_transformers.SentenceTransformer import SentenceTransformer as model_type


async def _setup_sidebar() -> None:
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


async def _grab_all_images(root_dir: str) -> list | None:
    """
    Recursively extracting all image path based on root path dir.

    Parameters:
    - root_path: Root directory for searching image data.

    Returns:
    - List of all image data in extention jpg, jpeg, png.
    """
    try:
        if not os.path.exists(path=root_dir):
            logging.error(
                f"[_grab_all_images] Directory {root_dir} not available, make sure its available in projects directory."
            )
            return None

        image_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = [
            str(path)
            for path in Path(root_dir).rglob("*")
            if path.suffix.lower() in image_extensions
        ]

        if not image_paths:
            logging.error(f"[_grab_all_images] No image files found in {root_dir}")
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
                f"[_normalize_embeddings] Expected embeddings to be Tensor or ndarray, got {type(embeddings)}"
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


async def _auto_update_encoding(
    cache_name: list, total_data_from_nas: int
) -> bool | None:
    try:
        if not cache_name:
            logging.info("[_auto_update_encoding] Initializing first encode.")
            return False

        total_current_data = int((cache_name[-1]).split("_")[-1].split(".")[0])
        diff = total_data_from_nas - total_current_data

        if diff >= 10:
            logging.info(
                "[_auto_update_encoding] Perform re-encode process due to triggered by additional data in NAS."
            )
            st.warning("Significant data change detected. Re-running encoding process.")
            return True

    except Exception as e:
        logging.info(
            f"[_auto_update_encoding] Error while performing auto update encoding: {e}"
        )
        return None
    return False


def _produce_image_queue(data: dict, queue_name: Literal["image_data"]) -> None:
    try:
        connection = BlockingConnection(ConnectionParameters(host=RABBITMQ_HOST))
        logging.info("[_produce_image_queue] Opened RabbitMQ connection.")

        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)

        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=data,
            properties=BasicProperties(delivery_mode=1),
        )
        logging.info(
            f"[_produce_text_queue] Text data prompt sent into {queue_name} queue."
        )

    except Exception as e:
        logging.error(f"[_produce_image_queue] Error while sending queue: {e}")
    finally:
        connection.close()
        logging.info("[_produce_image_queue] Closed RabbitMQ connection.")
    return None


def _produce_text_queue(data: dict, queue_name: Literal["text_data"]) -> None:
    try:
        connection = BlockingConnection(ConnectionParameters(host=RABBITMQ_HOST))
        logging.info("[_produce_text_queue] Opened RabbitMQ connection.")

        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)

        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=data,
            properties=BasicProperties(delivery_mode=1),
        )
        logging.info(
            f"[_produce_text_queue] Text data prompt sent into {queue_name} queue."
        )
    except Exception as e:
        logging.error(f"[_produce_text_queue] Error while sending queue: {e}")
    finally:
        connection.close()
        logging.info("[_produce_text_queue] Closed RabbitMQ connection.")
    return None


def _on_message_receives(model: model_type, channel, method, properties, body) -> None:
    try:
        logging.info(f"[Consumer] Received search query: {body}")

    except Exception as e:
        logging.error(f"[_on_message_receives] Error callback function: {e}")
    return None


def _consume_queue(queue_name: Literal["text_data", "image_data"]) -> None:
    try:
        connection = BlockingConnection(ConnectionParameters(host=RABBITMQ_HOST))
        logging.info("[_consume_queue] Opened RabbitMQ connection.")

        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        logging.info(f"[_consume_queue] Declaring queue into {queue_name} queue.")

        channel.basic_consume(
            queue=queue_name,
            auto_ack=True,
        )
    except Exception as e:
        logging.error(f"[_consume_queue] Error while sending queue: {e}")
    finally:
        connection.close()
        logging.info("[_consume_queue] Closed RabbitMQ connection.")
    return None


async def _wrapper_text_data(text_prompt: str, total_data: int) -> dict | None:
    try:
        data = TextData(text_prompt=text_prompt, total_data=total_data).dict()
        logging.info("[_wrapper_text_data] Created data TextData wrapper.")
    except Exception as e:
        logging.error(
            f"[_wrapper_text_data] Error while wrapping {data} into dictionary: {e}"
        )
        return None
    return data


# async def _wrapper_image_data(image_data: PILImage, total_data: int) -> dict|None:
#     try:
#         data = ImageData(image_data=image_data, total_data=total_data).dict()
#         logging.info("[_wrapper_image_data] Created data ImageData wrapper.")
#     except Exception as e:
#         logging.error(f"[_wrapper_image_data] Error while wrapping {data} into dictionary: {e}")
#         return None
#     return data
