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
from sentence_transformers.SentenceTransformer import SentenceTransformer as model_type
from utils.validator import (
    _check_gpu_memory,
    _check_already_have_encoded_data,
    _check_uploader,
    _check_gpu_availability,
)
from utils.helper import (
    _grab_all_images,
    _normalize_embeddings,
    _preprocess_image,
    _setup_sidebar,
    _auto_update_encoding,
    _search_data,
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

if "current_image_index" not in st.session_state:
    st.session_state["current_image_index"] = 0
if "next_button" not in st.session_state:
    st.session_state["next_button"] = False
if "prev_button" not in st.session_state:
    st.session_state["prev_button"] = False
if "similar_images" not in st.session_state:
    st.session_state["similar_images"] = None
if "current_uploaded_image" not in st.session_state:
    st.session_state["current_uploaded_image"] = None
if "current_text_query" not in st.session_state:
    st.session_state["current_text_query"] = None


@st.cache_resource
def init_model(
    model: str = "clip-ViT-B-32", device: str = "cpu"
) -> SentenceTransformer | None:
    """
    Load Visual Transformers model.
    This uses the CLIP model for encoding.

    |   Model 	        |   Top 1 Performance   |
    |   clip-ViT-B-32 	|   63.3                |
    |   clip-ViT-B-16 	|   68.1                |
    |   clip-ViT-L-14 	|   75.4                |
    """

    start_time = datetime.now()

    try:
        model = SentenceTransformer(model_name_or_path=model, device=device)
        end_time = datetime.now()
        logging.info(
            f"[init_model] Elapsed model initialization process: {end_time-start_time}"
        )
    except Exception as E:
        st.error(f"[init_model] Error loading CLIP model: {E}")
        return None
    return model


@st.cache_resource
def _encode_data(
    should_re_encode: bool,
    _model: model_type,
    image_paths: list,
    root_dir: str,
    cache_name: str,
    batch_size: int = 4,
) -> Tensor | None:
    """Encode and normalize data embeddings with preprocessing."""

    start_time = datetime.now()
    try:
        if not os.path.exists(path="cache"):
            logging.info("[_encode_data] Creating cache dir.")
            os.makedirs(name="cache", exist_ok=True)

        list_encoded_data = os.listdir(path="cache")
        similar_encoded_data = _check_already_have_encoded_data(
            root_dir=root_dir, encoded_list=list_encoded_data
        )

        if not should_re_encode and similar_encoded_data:
            logging.info(
                f"[_encode_data] Using cached {similar_encoded_data[-1]} encoding files."
            )
            encoded_data = np.load(f"cache/{similar_encoded_data[-1]}")

        if should_re_encode or not similar_encoded_data:
            logging.info(
                "[_encode_data] Re-create cache encoding files due to significant data changes in NAS or changed NAS mount directory."
            )
            processed_images = [
                _preprocess_image(Image.open(path)) for path in image_paths
            ]
            encoded_data = _model.encode(
                processed_images,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            cache_file = f"cache/{cache_name}.npy"
            np.save(cache_file, encoded_data.cpu().numpy())
            logging.info(f"[_encode_data] Created new encoding file {cache_file}.")

        end_time = datetime.now()
        logging.info(
            f"[_encode_data] Elapsed encoding data process: {end_time-start_time}"
        )
    except Exception as e:
        logging.error(f"[_encode_data] Error while encoding data: {e}")
        return None
    return _normalize_embeddings(encoded_data)


async def main() -> None:
    """
    Main function for streamlit run in asynchronous mode.
    """
    start_time = datetime.now()

    root_dir = "PREVIEW_IMAGE"
    method, is_using_filtered_year = await _setup_sidebar(
        root_path=f"mounted-nas-do-not-delete-data/{root_dir}"
    )

    is_using_cuda = await _check_gpu_availability()
    resource_usage = await _check_gpu_memory(is_cuda_available=is_using_cuda)
    model = init_model(
        device="cuda" if is_using_cuda and resource_usage < 0.75 else "cpu"
    )

    image_list = await _grab_all_images(
        root_dir=f"mounted-nas-do-not-delete-data/{root_dir}"
    )

    total_data = len(image_list)
    logging.info(f"[main] Total image data in NAS: {total_data}")

    list_encoded_data = os.listdir(path="cache")
    similar_encoded_data = _check_already_have_encoded_data(
        root_dir=root_dir, encoded_list=list_encoded_data
    )

    cache_name = f"encoded_data_{root_dir}_{total_data}"
    should_re_encode = await _auto_update_encoding(
        cache_name=similar_encoded_data, total_data_from_nas=total_data
    )

    encoded_data = (
        _encode_data(should_re_encode, model, image_list, root_dir, cache_name)
        if should_re_encode
        else _encode_data(False, model, image_list, root_dir, cache_name)
    )
    normalized_encoding = _normalize_embeddings(embeddings=encoded_data)

    try:
        if method == "Text Prompt":
            st.write("#### Search data by text")
            text_query = st.text_area(
                label="Input image description",
                help="Describe the detail of image you want to search.",
            )
            is_disabled = await _check_uploader(method=method, text_query=text_query)

        else:
            st.write("#### Search data by image")
            image_upload = st.file_uploader(
                label="Choose image file",
                help="Accepted only 1 image data with extensions such as 'jpeg', 'jpg', 'png'.",
                type=["jpeg", "jpg", "png"],
            )

            if image_upload:
                st.image(image=image_upload, width=500)
            is_disabled = await _check_uploader(
                method=method, image_upload=image_upload
            )

        search_button = st.button(
            label="Search",
            type="primary",
            help="Search current data into similar image.",
            disabled=is_disabled,
        )

        specific_year = (
            st.session_state.get("spesific_year") if is_using_filtered_year else None
        )
        if specific_year:
            logging.info(f"[main] Searching using specific filter {specific_year}.")

        total_retrieved_data = st.session_state.get("total_retrieve")

        if search_button:
            logging.info("[main] Perform image search.")

            query_embedding = (
                model.encode([text_query], convert_to_tensor=True)
                if method == "Text Prompt"
                else model.encode(
                    [_preprocess_image(Image.open(image_upload))],
                    convert_to_tensor=True,
                )
            )

            similar_images = await _search_data(
                query_emb=query_embedding,
                encoded_data=normalized_encoding,
                image_paths=image_list,
                return_data=total_retrieved_data,
                specific_year=specific_year,
            )

            if similar_images:
                st.session_state["similar_images"] = similar_images
                logging.info(
                    "[main] Similar images found. Use navigation to browse them."
                )
            else:
                st.error("No similar images found. Please perform a new search.")

        if st.session_state["similar_images"]:
            similar_images = st.session_state["similar_images"]
            current_image_index = st.session_state["current_image_index"]
            current_image, current_score = similar_images[current_image_index]

            st.write("### Detailed Image Preview")
            with st.expander("View Selected Image"):
                st.image(
                    image=current_image,
                    caption=f"Image path: {current_image}",
                    use_container_width=True,
                )

                prev_btn = st.button("Previous", key="prev_button")
                next_btn = st.button("Next", key="next_button")

                if prev_btn:
                    st.session_state.current_image_index = (
                        st.session_state.current_image_index - 1
                    ) % len(similar_images)

                if next_btn:
                    st.session_state.current_image_index = (
                        st.session_state.current_image_index + 1
                    ) % len(similar_images)

            st.write("### Similar Images Found")
            for i in range(0, len(similar_images), 3):
                cols = st.columns(3)
                for idx, (img_path, score) in enumerate(similar_images[i : i + 3]):
                    with cols[idx]:
                        st.image(
                            image=img_path,
                            use_container_width=True,
                            caption=f"Image path: {img_path}, Score: {score:.2f}",
                        )

            end_time = datetime.now()
            logging.info(
                f"[main] Finised search similar image estimate time: {end_time-start_time}"
            )

    except Exception as e:
        st.error(f"[main] Error while executing main file: {e}")
    return None


if __name__ == "__main__":
    asyncio.run(main())
