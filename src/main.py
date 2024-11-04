import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import asyncio
import torch
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from utils.logger import logging
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from utils.nas_connection import nas_connector
from src.secret import NAS_IP, NAS_PORT, NAS_USERNAME, NAS_PASSWORD

st.set_page_config(
    layout='wide',
    page_title='Dfactory Image Similarity Search'
)


if 'execute_using_cuda_cores' not in st.session_state:
    st.session_state['execute_using_cuda_cores'] = True

if 'cuda_memory' not in st.session_state:
    st.session_state['cuda_memory'] = 0

def grab_images_as_numpy(conn, share_name, folder_path, file_limit=10):
    """Fetch up to `file_limit` images from `folder_path` on NAS `share_name` and convert to NumPy arrays."""
    images = []
    try:
        file_list = conn.listPath(share_name, folder_path)
        for file in file_list:
            if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_obj = BytesIO()
                conn.retrieveFile(share_name, f"{folder_path}/{file.filename}", file_obj)
                file_obj.seek(0)
                image = Image.open(file_obj)
                images.append(np.array(image))  # Convert PIL Image to NumPy array
                if len(images) >= file_limit:
                    break
    except Exception as e:
        logging.error(f"Error fetching images: {e}")
    
    return images

def _check_gpu_memory(threshold: float = 0.75) -> str:
    """
    Check CUDA cores memory usage and return 
    cuda or cpu based on defined threshold. 
    """
    try:
        if torch.cuda.is_available():
            logging.info("CUDA cores available.")
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            usage_ratio = allocated_memory / total_memory

            if usage_ratio < threshold:
                return "cuda"
    except Exception as E:
        return f"Error while checking gpu memory: {E}"
    
    return "cpu"

device = _check_gpu_memory()

async def _check_multisearch() -> bool:
    """Checks if both search methods are used, returning True if so to disable the search button."""
    if st.session_state['image_description'] and st.session_state["image_uploader"]:
        logging.error("Error multi-method.")
        st.error("Error multi-method: Only one search method should be selected.")
        return True
    elif not st.session_state['image_description'] and not st.session_state["image_uploader"]:
        logging.warning("No data inputted.")
        st.warning("Warning: Please upload data or fill image description for performing image search.")
        return False
    st.success("Success input data.")
    return False


@st.cache_resource
def _load_vit_model(model: str = "clip-ViT-B-16") -> SentenceTransformer: 
    """
    Load Visual Transformers model. 
    You chould change model param into down bellow.

    Model 	        Top 1 Performance
    clip-ViT-B-32 	63.3
    clip-ViT-B-16 	68.1
    clip-ViT-L-14 	75.4
    """
    try:
        model = SentenceTransformer(model_name_or_path=model, device=device)
    except Exception as E:
        st.error(f'Error load_vit_model: {E}')
    return model


async def main():
    """
    Main function for streamlit run in asynchronous mode.
    """
    with st.sidebar:
        st.header('Image Similarity Search Engine')
        st.divider()
        st.subheader('Project Overview')
        st.write(
            '''
            This project demonstrates a ***Proof of Concept (PoC)*** for an advanced Image Similarity Search Engine tailored for private datasets. 
            Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images.
            '''
        )
        st.divider()
    
    col1, col2 =  st.columns(2)

    with col1:
        st.write("Search data by image")
        image_uploader = st.file_uploader(
            label="Choose image file",
            help="Accepted only 1 image data with extensions such as .jpeg, .jpg, .png",
            key="image_uploader",
            type=["jpeg", "jpg", "png"]
        )

    with col2:
        st.write("Search data by text")
        image_description = st.text_area(
            label="Input image description", 
            help="Describe the image you want to search.",
            key="image_description",
        )

    disable_search = await _check_multisearch()

    search_data = st.button(
        label='Search',
        type='primary',
        help='Search current data into similar image.',
        disabled=disable_search
    )
    
    st.write(st.session_state)
    # Initialize NAS connection and grab images if search button is pressed
    if search_data:
        conn = nas_connector(NAS_USERNAME, NAS_PASSWORD, NAS_IP)
        if conn:
            share_name = "Dfactory"
            folder_path = "/test_bastian/2017"
            image_arrays = grab_images_as_numpy(conn, share_name, folder_path)
            
            # Display images in Streamlit
            st.subheader("Images from NAS")
            for idx, img_array in enumerate(image_arrays):
                st.image(img_array, caption=f"Image {idx+1}", use_column_width=True)


if __name__ == '__main__':
    asyncio.run(main())