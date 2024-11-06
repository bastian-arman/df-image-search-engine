import os
from dotenv import load_dotenv

"""
Stored for all .env credentials.
"""

load_dotenv()

NAS_IP = os.getenv("NAS_IP")
NAS_PORT = os.getenv("NAS_PORT")
NAS_USERNAME = os.getenv("NAS_USERNAME")
NAS_PASSWORD = os.getenv("NAS_PASSWORD")
