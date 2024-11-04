import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.logger import logging
from synology_api.filestation import FileStation
from src.secret import NAS_IP, NAS_PORT, NAS_USERNAME, NAS_PASSWORD
from utils.helper import _grab_all_images
from pprint import pprint

def nas_connector(
    nas_ip: str = NAS_IP,
    nas_port: str = NAS_PORT,
    nas_username: str = NAS_USERNAME,
    nas_password: str = NAS_PASSWORD
) -> FileStation | str:
    try:
        conn = FileStation(
            ip_address=nas_ip,
            port=nas_port,
            username=nas_username,
            password=nas_password,
            secure=True
        )
        logging.info("NAS connected.")
    except Exception as E:
        return f"Error while connecting into NAS: {E}" 
    return conn


conn = nas_connector()
data = conn.get_file_list("/Dfactory/test_bastian/2017")
for data in data['data']['files']:
    if data["path"].endswith(("jpg", "png", "jpeg")):
        print(data['path'])