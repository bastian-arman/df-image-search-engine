import torch
import streamlit as st
from utils.logger import logging


async def _check_multisearch() -> bool:
    """Checks if both search methods are used, returning True if so to disable the search button."""
    if st.session_state.get("image_description") and st.session_state.get(
        "image_uploader"
    ):
        logging.error("Error multi-method.")
        st.error("Error multi-method: Only one search method should be selected.")
        return True
    elif not st.session_state.get("image_description") and not st.session_state.get(
        "image_uploader"
    ):
        logging.warning("No data inputted.")
        st.warning(
            "Warning: Please upload data or fill image description for performing image search."
        )
        return True
    elif st.session_state["image_description"].strip() == "" and st.session_state.get(
        "image_uploader"
    ):
        logging.info("Using upload image method.")
        st.success("Success input data.")
        return False
    elif st.session_state["image_description"].strip() == "":
        logging.error("Only whitespace detected in image description.")
        st.error("Error: Image description cannot contain only spaces.")
        return True
    st.success("Success input data.")
    return False


def _check_gpu_memory(threshold: float = 0.75) -> str | None:
    """
    Check CUDA cores memory usage and return
    cuda or cpu based on defined threshold.
    """
    device = "cpu"

    try:
        if torch.cuda.is_available():
            logging.info("CUDA cores available.")
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            usage_ratio = allocated_memory / total_memory

            if usage_ratio < threshold:
                device = "cuda"
            logging.info(f"Utilized cuda cores: {usage_ratio:.2f}")
    except Exception as E:
        logging.error(f"Error while checking gpu memory: {E}")
        return None
    return usage_ratio, device
