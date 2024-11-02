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

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, processor

@st.cache_resource
def load_vit_model(model: str = "clip-ViT-B-16") -> SentenceTransformer: 
    try:
        model = SentenceTransformer(model_name_or_path=model, device=device)
    except Exception as E:
        st.error(f"Error initializing vit model: {E}")
    return model

clip_model, processor = load_clip_model()

async def preprocess_incoming_data(incoming_data: list) -> list:
    try:
        images = []
        for file in incoming_data:
            image = Image.open(file).convert("RGB")
            images.append(image)
    except Exception as E:
        st.error(f"Error preprocessing incoming data: {E}")
    return images

async def predict_image_tag(images: list) -> list:
    try:
        # Prepare processor inputs with images and predefined labels as text
        inputs = processor(text=predefined_corpus, images=images, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        predictions = []
        for i, image_embed in enumerate(image_embeds):
            # Calculate cosine similarities for each image with all labels
            cos_scores = cosine_similarity(image_embed.unsqueeze(0), text_embeds).squeeze()
            # Get top 5 labels and their scores
            top5_indices = torch.topk(cos_scores, 5).indices
            top5_labels = [(predefined_corpus[idx], cos_scores[idx].item()) for idx in top5_indices]
            predictions.append(top5_labels)
        
        return predictions
    except Exception as E:
        st.error(f"Error predicting image tag: {E}")

async def main():
    st.title("Image Similarity Search Engine")
    
    with st.sidebar:
        initialize_method = st.selectbox(
            label="Select mode",
            options=("Search data", "Upload data")
        )
        
        with st.expander("See documentation"):
            if initialize_method == "Search data":
                st.write(
                    '''
                    This mode allows you to search for images based on similarity.
                    '''
                )
            else:
                st.write(
                    '''
                    In this mode, you can upload new images to add to the search dataset.
                    '''
                )

    if initialize_method == "Search data":
        st.text("Search data mode is activated.")
    else:
        new_data = st.file_uploader(
            label="Choose image file",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported for uploading data with single or multiple image files with extensions such as .jpeg, .jpg, .png",
        )

        if new_data:
            images = await preprocess_incoming_data(new_data)
            cols = st.columns(4)
            predictions = await predict_image_tag(images)
            for i, img in enumerate(images):
                with cols[i % 4]:
                    st.image(image=img, caption=new_data[i].name, use_column_width=True)
                    st_tags(
                        label="Predicted tags",
                        text="Press enter to add more tags",
                        value=[label for label, score in predictions[i]],
                        maxtags=10,
                        suggestions=["dakdjak", "dajkdaj"]
                    )   

if __name__ == "__main__":
    asyncio.run(main())
