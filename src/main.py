import sys
import os
import asyncio
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch import Tensor
from utils.logger import logging
from sentence_transformers import SentenceTransformer
from utils.validator import _check_multisearch, _check_gpu_memory, _update_device
from utils.helper import (
    _grab_all_images,
    _normalize_embeddings,
    _preprocess_image,
    _search_data,
    _setup_sidebar,
)

st.set_page_config(layout="wide", page_title="Dfactory Image Similarity Search")
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

root_dir = "PREVIEW_IMAGE"
start_time = datetime.now()
image_list = _grab_all_images(root_path=f"mounted-nas-do-not-delete-data/{root_dir}")
total_data = len(image_list)
cache_name = f"encoded_data_{root_dir}_{total_data}"

if "device" not in st.session_state:
    st.session_state["device"] = _check_gpu_memory()

if "total_initial_data" not in st.session_state:
    st.session_state["total_initial_data"] = total_data

# st.write(st.session_state)
# st.write(total_data)


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
        model = SentenceTransformer(
            model_name_or_path=model, device=st.session_state["device"]
        )
        end_time = datetime.now()
        logging.info(f"Elapsed model initialization process: {end_time-start_time}")
    except Exception as E:
        st.error(f"Error loading CLIP model: {E}")
        return None
    return model


if _update_device():
    model = init_model()
else:
    model = init_model()


@st.cache_data
def _encode_data(image_paths: list, batch_size: int = 4) -> Tensor | None:
    """Encode and normalize data embeddings with preprocessing."""
    try:
        if not os.path.exists(path="cache"):
            logging.info("Creating cache dir.")
            os.makedirs(name="cache", exist_ok=True)

        cache_file = f"cache/{cache_name}.npy"

        if Path(cache_file).exists():
            logging.info("Cache files already created.")
            encoded_data = np.load(cache_file)
        else:
            logging.info("Creating cache encoding files.")
            processed_images = [
                _preprocess_image(Image.open(path)) for path in image_paths
            ]
            encoded_data = model.encode(
                processed_images,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            np.save(cache_file, encoded_data.cpu().numpy())
        end_time = datetime.now()
        logging.info(f"Elapsed encoding data process: {end_time-start_time}")
    except Exception as e:
        logging.error(f"[_encode_data] Error while encoding data: {e}")
        return None
    return _normalize_embeddings(encoded_data)


def _auto_update_encoding() -> Tensor | None:
    total_current_data = int(cache_name.split("_")[-1])
    diff = total_current_data - st.session_state["total_initial_data"]
    if diff == 10:
        st.warning("Significant data change detected. Re-running encoding process.")
        encoded_data = _encode_data(image_paths=image_list)
        st.session_state["current_data"] = total_data

    else:
        encoded_data = _encode_data(image_paths=image_list)
    return encoded_data


encoded_data = _auto_update_encoding()
normalized_encoding = _normalize_embeddings(embeddings=encoded_data)


async def main() -> None:
    """
    Main function for streamlit run in asynchronous mode.
    """
    try:
        _setup_sidebar()
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

        row_input = st.columns((0.9, 2, 2, 1))
        with row_input[0]:
            num_results = st.number_input(
                label="Total retrieve data",
                min_value=1,
                max_value=100,
                value=10,
                help="Input total of image that you want to retrieve, accepted with minimum value is 1 and maximum value as 100.",
                key="total_retrieve",
            )

        disable_search = await _check_multisearch()

        if image_file and not disable_search:
            st.image(image=image_file, width=500)

        search_button = st.button(
            label="Search",
            type="primary",
            help="Search current data into similar image.",
            disabled=disable_search,
        )

        if search_button:
            if image_file:
                query_image = _preprocess_image(Image.open(image_file))
                query_emb = model.encode([query_image], convert_to_tensor=True)
            else:
                query_emb = model.encode([text_query], convert_to_tensor=True)

            with st.spinner("Searching for similar images..."):
                similar_images = await _search_data(
                    query_emb=query_emb,
                    encoded_data=normalized_encoding,
                    image_paths=image_list,
                    return_data=num_results,
                )

            if similar_images:
                st.write("### Similar Images Found")
                for i in range(0, len(similar_images), 4):
                    cols = st.columns(4)
                    for idx, (img_path, score) in enumerate(similar_images[i : i + 4]):
                        with cols[idx]:
                            st.image(
                                image=img_path,
                                use_container_width=True,
                                caption=f"Image path: {img_path}",
                            )
                end_time = datetime.now()
                logging.info(f"Elapsed time for search data: {end_time-start_time}")
            else:
                st.error("No similar images found.")
    except Exception as e:
        st.error(f"[main] Error while executing main file: {e}")
    return None


if __name__ == "__main__":
    asyncio.run(main())
