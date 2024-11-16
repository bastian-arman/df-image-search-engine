import os
import sys
import pika
import json
import random
import string
import streamlit as st
from numpy import ndarray
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from numpy import isnan
from torch import Tensor, tensor
from typing import Literal
from numpy.linalg import norm
from PIL import ImageOps, Image
from utils.logger import logging
from utils.schema import QueueData
from src.secret import RABBITMQ_HOST
from sentence_transformers import util
from PIL.Image import Image as PILImage
from pika.exceptions import AMQPConnectionError
from pika import BlockingConnection, ConnectionParameters, BasicProperties


async def _setup_sidebar(root_path: str) -> tuple[str, bool] | None:
    """
    Create sidebar project description.
    """
    try:
        with st.sidebar:
            st.header("Image Similarity Search Engine")
            st.divider()

            method = st.selectbox(
                label="Select searching method",
                options=("Image Uploader", "Text Prompt"),
                help="Select searching method, only available Image Uploader or Text Prompt.",
            )

            st.number_input(
                label="Total retrieve data",
                min_value=1,
                max_value=100,
                value=10,
                help="Input total of image that you want to retrieve, accepted with minimum value is 1 and maximum value as 100.",
                key="total_retrieve",
            )

            is_using_filtered_year = st.toggle("Enable filter data source")
            options = sorted(
                [year for year in os.listdir(path=root_path) if len(year) == 4]
            )

            if is_using_filtered_year:
                st.selectbox(
                    label="Select data source",
                    options=options,
                    help="Focus data search on spesific years of data.",
                    key="spesific_year",
                    index=(len(options) - 1),
                )

            st.divider()
            st.subheader(body="Project Overview")
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
    return method, is_using_filtered_year


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
        image = ImageOps.flip(image)
        image = ImageOps.expand(image)
        image = ImageOps.mirror(image)
    except Exception as e:
        logging.error(f"[_preprocess_image] Error while preprocessing an image: {e}")
        return None
    return image


def _normalize_embeddings(embeddings: Tensor | ndarray) -> ndarray | None:
    """
    Normalize embedding tensor from range 1 into 0

    Parameters:
    - embeddings: Tensor or ndarray to be normalized

    Returns:
    - Normalized embeddings as ndarray, or None if an error occurs.
    """
    try:
        if not isinstance(embeddings, (Tensor, ndarray)):
            logging.error(
                f"[_normalize_embeddings] Expected embeddings to be Tensor or ndarray, got {type(embeddings)}"
            )
            return None

        if isinstance(embeddings, Tensor) and embeddings.device.type == "cuda":
            embeddings = embeddings.cpu()

        embedding_data = (
            embeddings.detach().numpy()
            if isinstance(embeddings, Tensor)
            else embeddings
        )

        epsilon = 1e-8
        embedding_norms = norm(embedding_data, axis=1, keepdims=True) + epsilon
        normalized_embeddings = embedding_data / embedding_norms

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
    specific_year=None,
) -> list | None:
    """
    Search for similar images based on a precomputed query embedding, with an optional year filter.

    Parameters:
    - query_emb: The embedding of the query (text or image).
    - encoded_data: The embeddings of all images in the database.
    - image_paths: The paths of all images in the database, corresponding to the embeddings.
    - return_data: Number of top similar images to return.
    - specific_year: Filter results based on the year, if provided.

    Returns:
    - List of tuples with the image path and similarity score for the top `return_data` similar images.
    """
    try:
        query_emb = _normalize_embeddings(embeddings=query_emb)
        encoded_data = _normalize_embeddings(embeddings=encoded_data)

        if specific_year:
            logging.info(f"[_search_data] Using {specific_year} filter.")
            year_filtered_paths = [
                path
                for path in image_paths
                if str(specific_year) in os.path.basename(path)
            ]

            root_dir = year_filtered_paths[0].split("/")[1]
            current_dir_data = len(
                os.listdir(f"mounted-nas-do-not-delete-data/{root_dir}/{specific_year}")
            )

            if not year_filtered_paths:
                st.warning(f"No images found for the year {specific_year}.")
                return None

            filtered_indices = [
                idx
                for idx, path in enumerate(image_paths)
                if path in year_filtered_paths
            ]
            filtered_encoded_data = encoded_data[filtered_indices]
        else:
            year_filtered_paths = image_paths
            filtered_encoded_data = encoded_data

        hits = util.semantic_search(
            query_embeddings=query_emb,
            corpus_embeddings=filtered_encoded_data,
            top_k=return_data,
        )[0]

        similar_images = [
            (year_filtered_paths[hit["corpus_id"]], hit["score"]) for hit in hits
        ]

        if len(similar_images) < return_data:
            st.warning(
                f"Selected year have {current_dir_data} data, but only {len(similar_images)} similar images found for the selected year."
            )

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
    """
    Trigger auto re-encode when there is more of 10 images data update in NAS.

    Parameters:
    - cache_name: List of already saved cache in /cache directory.
    - total_data_from_nas: Total image data from NAS.

    Returns:
    - Boolean for auto triggering update encoding process.
    """
    try:
        if not cache_name:
            logging.info("[_auto_update_encoding] Initializing first encode.")
            return False

        total_current_data = int((cache_name[-1]).split("_")[-1].split(".")[0])
        diff = total_data_from_nas - total_current_data

        if total_data_from_nas <= total_current_data:
            logging.info(
                f"[_auto_update_encoding] Skipping re-encode. Current encoded data is {total_current_data} while incoming data is {total_data_from_nas}"
            )
            return False

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


async def _wrapper_queue_data(
    query_embedding: list[float], total_retrieved_data: int
) -> dict | None:
    try:
        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            logging.error(
                f"[_wrapper_queue_data] Invalid query_embedding: {query_embedding}"
            )
            return None

        if not isinstance(total_retrieved_data, int) or total_retrieved_data < 0:
            logging.error(
                f"[_wrapper_queue_data] Invalid total_retrieved_data: {total_retrieved_data}"
            )
            return None

        data = QueueData(
            query_embedding=query_embedding, total_retrieved_data=total_retrieved_data
        ).model_dump()
        logging.info("[_wrapper_queue_data] Created data QueueData wrapper.")
    except Exception as e:
        logging.error(
            f"[_wrapper_queue_data] Error while wrapping QueueData into dictionary: {e}"
        )
        return None

    return data


async def _produce_queue(
    data: dict, queue_name: Literal["text_query", "image_upload"]
) -> None:
    try:
        connection = BlockingConnection(ConnectionParameters(host=RABBITMQ_HOST))
        logging.info("[_produce_queue] Opened RabbitMQ connection.")

        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)

        message_body = json.dumps(data)

        channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=message_body,
            properties=BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
        )
        logging.info(f"[_produce_queue] Text data prompt sent into {queue_name} queue.")
    except AMQPConnectionError as e:
        logging.error(f"[_produce_queue] Connection to RabbitMQ failed: {e}")
        raise ConnectionError(
            "RabbitMQ container may not be running. Please ensure it is started."
        )
    except Exception as e:
        logging.error(f"[_produce_queue] Error while sending queue: {e}")
    finally:
        connection.close()
        logging.info("[_produce_queue] Closed RabbitMQ connection.")
    return None


async def _consume_queue(
    queue_name: Literal["text_query", "image_upload"],
    image_list: list,
    encoding: ndarray,
) -> list | None:
    result_image_paths = None

    async def callback(ch, method, properties, body):
        nonlocal result_image_paths
        try:
            message = json.loads(body)
            query_embedding = tensor(message["query_embedding"]).unsqueeze(0)
            total_retrieved_data = message["total_retrieved_data"]

            similar_images = await _search_data(
                query_emb=query_embedding,
                encoded_data=encoding,
                image_paths=image_list,
                return_data=total_retrieved_data,
            )

            if similar_images:
                result_image_paths = similar_images
                logging.info(f"Found {len(result_image_paths)} similar images.")
            else:
                logging.warning("No similar images found.")
                result_image_paths = None

            ch.basic_ack(delivery_tag=method.delivery_tag)
            channel.stop_consuming()  # Stop after processing one message

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            ch.basic_ack(
                delivery_tag=method.delivery_tag
            )  # Acknowledge to remove from queue even on error

    try:
        connection = BlockingConnection(ConnectionParameters(host=RABBITMQ_HOST))
        logging.info("[_consume_queue] Opened RabbitMQ connection.")

        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=queue_name, on_message_callback=await callback)
        logging.info(f"[_consume_queue] Waiting for messages in {queue_name}")

        channel.start_consuming()

    except AMQPConnectionError as e:
        logging.error(f"[_consume_queue] RabbitMQ connection failed: {e}")
        raise ConnectionError(
            "RabbitMQ container may not be running. Please ensure it is started."
        )
    except Exception as e:
        logging.error(
            f"[_consume_queue] Error while consuming from queue {queue_name}: {e}"
        )
    finally:
        connection.close()
        logging.info("[_consume_queue] Closed RabbitMQ connection.")

    return result_image_paths
