import sys
import numpy as np
from PIL import ImageOps, Image
from PIL.Image import Image as PILImage
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import asyncio
import streamlit as st
from torch import Tensor
from utils.logger import logging
from sentence_transformers import SentenceTransformer, util
from utils.validator import _check_multisearch, _check_gpu_memory
from utils.helper import _grab_all_images

st.set_page_config(layout="wide", page_title="Dfactory Image Similarity Search")

# TODO: encapsulate each function for easier to debug when error occured.

st.markdown(
    """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton  {visibility:hidden;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""",
    unsafe_allow_html=True,
)

gpu_usage, device = _check_gpu_memory()

if "execute_using_cuda_cores" not in st.session_state:
    st.session_state["execute_using_cuda_cores"] = True

if "cuda_memory" not in st.session_state:
    st.session_state["cuda_memory"] = gpu_usage


st.write(st.session_state)


@st.cache_resource
def init_model(model: str = "clip-ViT-B-32") -> SentenceTransformer | None:
    """
    Load Visual Transformers model.
    This uses the CLIP model for encoding.

    |   Model 	        |   Top 1 Performance   |
    |   clip-ViT-B-32 	|   63.3                |
    |   clip-ViT-B-16 	|   68.1                |
    |   clip-ViT-L-14 	|   75.4                |
    """
    try:
        model = SentenceTransformer(model_name_or_path=model, device=device)
    except Exception as E:
        st.error(f"Error loading CLIP model: {E}")
        return None
    return model


model = init_model()
image_list = _grab_all_images(root_path="mounted-nas-do-not-delete-data/test_bastian")


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit length, moving to CPU if necessary."""
    if isinstance(embeddings, Tensor):
        embeddings = (
            embeddings.cpu().numpy()
        )  # Move to CPU and convert to NumPy if it's a Tensor
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm


def preprocess_image(image: PILImage) -> PILImage:
    """Preprocess image by resizing, grayscaling, and normalizing."""
    image = ImageOps.fit(image, (224, 224))
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageOps.invert(image)
    image = ImageOps.mirror(image)
    return image


@st.cache_data
def encode_data(_images: list):
    """Encode and normalize data embeddings with preprocessing."""
    processed_images = [preprocess_image(Image.open(path)) for path in _images]
    encoded_data = model.encode(
        processed_images,
        batch_size=128,
        convert_to_tensor=False,
        show_progress_bar=True,
    )
    return normalize_embeddings(encoded_data)


encoded_data = encode_data(_images=image_list)


async def _search_data(
    query_emb: Tensor,
    encoded_data: Tensor,
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
        # Normalize embeddings
        query_emb = normalize_embeddings(query_emb)
        encoded_data = normalize_embeddings(encoded_data)
        # Perform semantic search
        hits = util.semantic_search(query_emb, encoded_data, top_k=return_data)[0]
        # Retrieve similar image paths and scores
        st.write(image_paths[hits[0]["corpus_id"]])
        similar_images = [(image_paths[hit["corpus_id"]], hit["score"]) for hit in hits]
    except Exception as e:
        logging.error(f"[_search_data] Error while searching data: {e}")
        return None
    return similar_images


async def main():
    """
    Main function for streamlit run in asynchronous mode.
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

        col1, col2 = st.columns(2)

        with col1:
            st.write("Search data by image")
            image_file = st.file_uploader(
                label="Choose image file",
                help="Accepted only 1 image data with extensions such as 'jpeg', 'jpg', 'png'.",
                key="image_uploader",
                type=["jpeg", "jpg", "png"],
            )

        with col2:
            st.write("Search data by text")
            text_query = st.text_area(
                label="Input image description",
                help="Describe the detail of image you want to search.",
                key="image_description",
            )

        row_input = st.columns((1, 2, 2, 1))
        with row_input[0]:
            num_results = st.number_input(
                label="Total extracted data",
                min_value=1,
                max_value=100,
                value=5,
                help="Total image retrieve data.",
                key="total_retrieve",
            )

        disable_search = await _check_multisearch()
        if image_file:
            st.image(image=image_file)

        search_button = st.button(
            label="Search",
            type="primary",
            help="Search current data into similar image.",
            disabled=disable_search,
        )

        if search_button:
            if image_file:
                # Preprocess and encode the image
                query_image = preprocess_image(Image.open(image_file))
                query_emb = model.encode([query_image], convert_to_tensor=True)
            elif text_query:
                # Encode the text query
                query_emb = model.encode([text_query], convert_to_tensor=True)
            else:
                st.warning("Please provide either an image or a text description.")
                return

            # Perform the search
            similar_images = await _search_data(
                # model=model,
                query_emb=query_emb,
                encoded_data=encoded_data,
                image_paths=image_list,
                return_data=num_results,
            )

            # Display results
            if similar_images:
                st.write(similar_images)
                st.write("### Similar Images Found")
                for image_path, score in similar_images:
                    st.image(
                        image_path,
                        caption=f"Score: {score:.4f}\nImage path : {image_path}",
                    )
            else:
                st.error("No similar images found.")
    except Exception as e:
        st.error(f"[main] Error while executing main file: {e}")

    # finally:
    #     nas_connection.close()
    #     logging.info("Closed SMB connection.")


if __name__ == "__main__":
    asyncio.run(main())
