import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import asyncio
import torch
import streamlit as st
from PIL import Image
from utils.logger import logging
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from utils.nas_connection import nas_connector

st.set_page_config(
    layout='wide',
    page_title='Dfactory Image Similarity Search'
)


if 'execute_using_cuda_cores' not in st.session_state:
    st.session_state['execute_using_cuda_cores'] = True

if 'cuda_memory' not in st.session_state:
    st.session_state['cuda_memory'] = 0


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
            Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images, 
            making it ideal for applications in fields such as media management, content moderation, and digital asset organization.
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
    # st.write(image_description)


if __name__ == '__main__':
    asyncio.run(main())