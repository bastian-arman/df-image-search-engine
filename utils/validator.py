import sys
import torch
import streamlit as st
from pathlib import Path
from utils.logger import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))


async def _check_multisearch(
    image_description: str = None, image_uploader: str = None
) -> bool:
    """Checks if both search methods are used, returning True if so to disable the search button."""
    description = (
        image_description
        if image_description is not None
        else st.session_state.get("image_description")
    )
    uploader = (
        image_uploader
        if image_uploader is not None
        else st.session_state.get("image_uploader")
    )

    try:
        if description and uploader:
            logging.error("Error multi-method.")
            st.error("Error multi-method: Only one search method should be selected.")
            return True
        if not description and not uploader:
            logging.warning("No data inputted.")
            st.warning(
                "Warning: Please upload data or fill image description to perform image search."
            )
            return True
        if description and description.strip() == "" and uploader:
            logging.info("Using upload image method.")
            st.success("Success input data.")
            return False
        if description and description.strip() == "":
            logging.error("Only whitespace detected in image description.")
            st.error("Error: Image description cannot contain only spaces.")
            return True
    except Exception as e:
        logging.error(
            f"[_check_multisearch] Error exception while check multisearch: {e}"
        )
        return None
    st.success("Success input data.")
    return False


async def _check_gpu_avaibility() -> bool | None:
    try:
        if torch.cuda.is_available():
            logging.info("[_check_gpu_avaibility] CUDA cores available.")
            return True
    except Exception as e:
        logging.info(
            f"[_check_gpu_avaibility] Error while checking gpu avaibility: {e}"
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
        usage_ratio = allocated_memory / total_memory

        logging.info(f"[_check_gpu_memory] Utilized cuda cores: {usage_ratio:.2f}")
    except Exception as E:
        logging.error(f"[_check_gpu_memory] Error while checking GPU memory: {E}")
        return None
    return usage_ratio


def _check_already_have_encoded_data(root_dir: str, encoded_list: list) -> list | None:
    try:
        matching_data = sorted(
            [
                item
                for item in encoded_list
                if root_dir in item.split("encoded_data_")[1]
            ]
        )

        if not matching_data:
            logging.error(
                "[_check_already_have_encoded_data] No similar encoder find. Should initialize encode data."
            )
            return None

        if not encoded_list:
            logging.error(
                "[_check_already_have_encoded_data] No list of encoded data found! Please check the /cache directory.."
            )
            return None

    except Exception as e:
        logging.error(
            f"[_check_already_have_encoded_data] Error while checking if any already saved encoder data: {e}"
        )
        return None
    return matching_data
