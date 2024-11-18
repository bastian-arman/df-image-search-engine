import sys
import torch
import streamlit as st
from typing import Literal
from pathlib import Path
from utils.logger import logging
from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append(str(Path(__file__).resolve().parents[1]))


async def _check_uploader(
    method: Literal["Image Uploader", "Text Prompt"],
    image_upload: UploadedFile = None,
    text_query: str = None,
) -> bool:
    if method == "Text Prompt" and not text_query:
        logging.warning("No data inputted.")
        st.warning("Warning: Please fill image description to perform image search.")
        st.session_state["similar_images"] = None
        st.session_state["current_image_index"] = 0
        return True
    if method == "Image Uploader" and not image_upload:
        logging.warning("No data inputted.")
        st.warning("Warning: Please upload data to perform image search.")
        st.session_state["similar_images"] = None
        st.session_state["current_image_index"] = 0
        return True
    if method == "Text Prompt" and text_query.strip() == "":
        logging.error("Only whitespace detected in image description.")
        logging.info("Image description cannot contain only spaces.")
        st.error("Error: Image description cannot contain only spaces.")
        st.session_state["similar_images"] = None
        st.session_state["current_image_index"] = 0
        return True
    if method == "Text Prompt" and text_query.strip() != "":
        current_text = st.session_state.get("current_text_query", None)
        if current_text != text_query.strip():
            logging.info("New text prompt detected. Resetting session state.")
            st.session_state["similar_images"] = None
            st.session_state["current_image_index"] = 0
        st.session_state["current_text_query"] = text_query.strip()
        st.success("Success input text data.")
        return False

    if method == "Image Uploader" and image_upload:
        current_image = st.session_state.get("current_uploaded_image", None)
        if current_image != image_upload.name:
            logging.info("New image uploaded. Resetting session state.")
            st.session_state["similar_images"] = None
            st.session_state["current_image_index"] = 0
        st.session_state["current_uploaded_image"] = image_upload.name
        st.success("Success uploading image data.")
        return False
    return True


async def _check_gpu_availability() -> bool | None:
    try:
        if torch.cuda.is_available():
            logging.info("[_check_gpu_availability] CUDA cores available.")
            if torch.cuda.device_count() > 0:
                if torch.cuda.get_device_properties(0).total_memory > 0:
                    logging.info(
                        "[_check_gpu_availability] GPU device and memory available."
                    )
                    return True
                else:
                    logging.warning(
                        "[_check_gpu_availability] GPU device available but no memory."
                    )
                    return False
            else:
                logging.warning("[_check_gpu_availability] No CUDA devices found.")
                return False
    except Exception as e:
        logging.info(
            f"[_check_gpu_availability] Error while checking GPU availability: {e}"
        )
        return None
    return False


async def _check_gpu_memory(is_cuda_available: bool) -> float | None:
    """
    Parameters:
    - is_cuda_available: CUDA cores is available.

    Returns:
    - Float of CUDA cores memory usage in percentage.
    """
    try:
        if not isinstance(is_cuda_available, bool):
            logging.error("[_check_gpu_memory] Invalid type for is_cuda_available.")
            return None

        if not is_cuda_available:
            logging.warning(
                "[_check_gpu_memory] Warning, CUDA cores not available in this machine, proceeding execution code into CPU device."
            )
            return None

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)

        if total_memory <= 0:
            logging.error(
                "[_check_gpu_memory] Total memory reported as zero or negative."
            )
            return None

        if allocated_memory > total_memory:
            logging.warning(
                "[_check_gpu_memory] Allocated memory exceeds total memory."
            )
            return None

        usage_ratio = allocated_memory / total_memory
        logging.info(f"[_check_gpu_memory] CUDA memory usage: {usage_ratio:.2f}")
    except Exception as E:
        logging.error(f"[_check_gpu_memory] Error while checking GPU memory: {E}")
        return None
    return usage_ratio


def _check_already_have_encoded_data(root_dir: str, encoded_list: list) -> list | None:
    try:
        if not root_dir and not encoded_list:
            logging.error(
                "[_check_already_have_encoded_data] Provided empty encoded list and empty root directory!"
            )
            return None

        if not encoded_list:
            logging.error(
                "[_check_already_have_encoded_data] No list of encoded data found! Please check the /cache directory."
            )
            return None

        if not root_dir:
            logging.error(
                "[_check_already_have_encoded_data] No root directory data found! Root directory should be filled with NAS directory."
            )
            return None

        matching_data = sorted(
            [
                item
                for item in encoded_list
                if root_dir in item.split("encoded_data_")[1]
            ]
        )

        if not matching_data:
            logging.warning(
                "[_check_already_have_encoded_data] No similar encoder found. Should initialize encode data."
            )
            return None

    except Exception as e:
        logging.error(
            f"[_check_already_have_encoded_data] Error while checking if any already saved encoder data: {e}"
        )
        return None
    return matching_data
