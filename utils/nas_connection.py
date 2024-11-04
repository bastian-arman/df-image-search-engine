import sys
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from PIL import Image
from smb.SMBConnection import SMBConnection
from io import BytesIO
from src.secret import NAS_IP, NAS_PORT, NAS_USERNAME, NAS_PASSWORD
from utils.logger import logging

def nas_connector(username: str, password: str, nas_ip: str) -> SMBConnection:
    try:
        conn = SMBConnection(
            username=username,
            password=password,
            my_name="",
            remote_name="",  # Set this to your NAS name if necessary
            use_ntlm_v2=True
        )
        conn.connect(nas_ip, 139)  # Use port 139 or 445 depending on NAS configuration
        logging.info("Connected to NAS.")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to NAS: {e}")
        return None

def grab_images_as_numpy(conn, share_name, folder_path, file_limit=10):
    """
    Fetch up to `file_limit` images from `folder_path` on NAS `share_name` and convert to NumPy arrays.
    """
    images = []
    try:
        # List files in the directory
        file_list = conn.listPath(share_name, folder_path)
        
        for file in file_list:
            if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Retrieve the file content
                file_obj = BytesIO()
                conn.retrieveFile(share_name, f"{folder_path}/{file.filename}", file_obj)
                
                # Convert to PIL Image, then to NumPy array
                file_obj.seek(0)
                image = Image.open(file_obj)
                images.append(np.array(image))  # Convert PIL Image to NumPy array

                if len(images) >= file_limit:
                    break
    except Exception as e:
        logging.error(f"Error fetching images: {e}")
    
    return images

# Initialize NAS connection
conn = nas_connector(NAS_USERNAME, NAS_PASSWORD, NAS_IP)

# Define NAS share name and directory path
share_name = "Dfactory"
folder_path = "/test_bastian/2017"

# Grab the first 10 image files as NumPy arrays
image_arrays = grab_images_as_numpy(conn, share_name, folder_path)

# Print information about retrieved images
for idx, img_array in enumerate(image_arrays):
    print(f"Image {idx+1}: Shape={img_array.shape}, Dtype={img_array.dtype}")
