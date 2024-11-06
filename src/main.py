import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import asyncio
import streamlit as st
from utils.logger import logging
from sentence_transformers import SentenceTransformer
from utils.nas_connection import nas_connector
from utils.validator import _check_multisearch, _check_gpu_memory

#: TODO find way to retrive all data directly from NAS
# """
# 1. Grab all dir inside the root_dir (loop recursively so all dir can be found) (DONE)
# 2. Grab all image data inside (DONE)
# """
#: TODO create preprocessing data pipeline
# """
# Streamline all input data into same size and same preprocessed data (e.g: Threshold, Flip, etc) (DONE)
# """

#: TODO create image search pipeline for finding similar images directly from nas

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

if "execute_using_cuda_cores" not in st.session_state:
    st.session_state["execute_using_cuda_cores"] = True

if "cuda_memory" not in st.session_state:
    st.session_state["cuda_memory"] = 0

device = _check_gpu_memory()
service_name = "Dfactory"
root_dir = "test_bastian"


@st.cache_resource
def init_model(model: str = "clip-ViT-B-32") -> SentenceTransformer:
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
    return model


model = init_model()


async def main():
    """
    Main function for streamlit run in asynchronous mode.
    """
    try:
        nas_connection = await nas_connector()
        # dirs = await _directory_finder(nas_connector=nas_connection, service_name=service_name, root_dir=root_dir)
        # image_paths = await _image_finder(nas_connector=nas_connection, service_name=service_name, image_dirs=dirs)

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
            st.file_uploader(
                label="Choose image file",
                help="Accepted only 1 image data with extensions such as 'jpeg', 'jpg', 'png'.",
                key="image_uploader",
                type=["jpeg", "jpg", "png"],
            )

        with col2:
            st.write("Search data by text")
            st.text_area(
                label="Input image description",
                help="Describe the detail of image you want to search.",
                key="image_description",
            )

        row_input = st.columns((1, 2, 2, 1))
        with row_input[0]:
            st.number_input(
                label="Total extracted data",
                min_value=1,
                max_value=100,
                value=5,
                help="Total image retrieve data.",
                key="total_retrieve",
            )

        disable_search = await _check_multisearch()

        st.button(
            label="Search",
            type="primary",
            help="Search current data into similar image.",
            disabled=disable_search,
        )

        # if search_data:
        #     _search_data(model=model, query=)
    except Exception as e:
        st.error(f"[main] Error while executing main file: {e}")

    finally:
        nas_connection.close()
        logging.info("Closed SMB connection.")


if __name__ == "__main__":
    asyncio.run(main())
