import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import asyncio
import torch
import streamlit as st
from PIL import Image
from torch import Tensor
from utils.corpus import predefined_corpus
from src.database.connection import db_connection
from streamlit_tags import st_tags
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    layout="wide",
    page_title="Dfactory Image Similarity Search"
)


if "safe_to_upload" not in st.session_state:
    st.session_state.safe_to_upload = False
# st.write(st.session_state)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

@st.cache_resource
def load_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as E:
        st.error(f"Error load_clip_model: {E}")
    return clip_model, processor

@st.cache_resource
def load_vit_model() -> SentenceTransformer: 
    try:
        model = SentenceTransformer(model_name_or_path="clip-ViT-B-16", device=device)
    except Exception as E:
        st.error(f"Error load_vit_model: {E}")
    return model

clip_model, processor = load_clip_model()

def validate_total_files(data: list) -> None:
    accepted_files = 5
    total_files = len(data)

    if total_files == 0:
        st.session_state.safe_to_upload = False
        st.warning("Please upload your data.")
    elif total_files <= accepted_files:
        st.session_state.safe_to_upload = True
        st.success("File successfully uploaded.")
    else:
        excess_files = total_files - accepted_files
        st.session_state.safe_to_upload = False
        st.error(f"Max file accepted is {accepted_files}. "
                 f"You have uploaded {total_files} files. "
                 f"Please delete {excess_files} file(s) to proceed.")

async def preprocess_incoming_data(incoming_data: list) -> list:
    images = []
    for file in incoming_data:
        image = Image.open(file).convert("RGB")
        images.append(image)
    return images

async def predict_image_tag(images: list) -> list:
    try:
        inputs = processor(text=predefined_corpus, images=images, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        predictions = []
        for i, image_embed in enumerate(image_embeds):
            cos_scores = cosine_similarity(image_embed.unsqueeze(0), text_embeds).squeeze()
            top5_indices = torch.topk(cos_scores, 5).indices
            top5_labels = [(predefined_corpus[idx], cos_scores[idx].item()) for idx in top5_indices]
            predictions.append(top5_labels)
        
        return predictions
    except Exception as E:
        st.error(f"Error predicting image tag: {E}")

async def main():
    with st.sidebar:
        st.header("Image Similarity Search Engine")
        initialize_method = st.selectbox(
            label="Select mode",
            options=("Search data", "Upload data")
        )
        
        with st.expander("See documentation"):
            if initialize_method == "Search data":
                st.write("In Search mode, you can find images by either describing what you're looking for or by uploading an image that resembles the one you want to find. This helps you quickly locate visually similar images.")
            else:
                st.write("In Upload mode, you can add new images to our collection. These images will be saved in the database, helping to create a richer dataset that improves search results in the future.")



        st.divider()
        st.subheader("Project Overview")
        st.write(
            """
            This project demonstrates a **Proof of Concept (PoC)** for an advanced Image Similarity Search Engine tailored for private datasets. 
            Designed to search and retrieve visually similar images, this project leverages state-of-the-art models to efficiently index and query images, 
            making it ideal for applications in fields such as media management, content moderation, and digital asset organization.
            """
        )
        st.divider()

    if initialize_method == "Search data":
        image_description = st.text_area(label="Input image description")
    else:
        upload_data = st.file_uploader(
            label="Choose image file",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported for uploading data with single or multiple image files with extensions such as .jpeg, .jpg, .png"
        )

        validate_total_files(data=upload_data)

        if st.session_state.safe_to_upload and upload_data:
            search_by_keyword = st.checkbox(
                label="Activate search by keyword",
                value=True
            )
            save_data = st.button(
                label="Save data",
                type="primary",
                help="Save tagged image into database."
            )

            cols = st.columns(4)
            for i, img in enumerate(upload_data):
                with cols[i % 4]:
                    st.image(image=img, caption=img.name, use_column_width=True)

if __name__ == "__main__":
    asyncio.run(main())
