#!/bin/sh

# Load variables from .env
if [ -f "$PWD/.env" ]; then
    echo "Found .env file!"
    . "$PWD/.env"
else
    echo ".env File not found!"
    exit 1
fi

# Create new dir, skip if it already exists
mkdir -p mounted-nas-do-not-delete-data

# Get the current directory path
current_pwd=$(pwd)

# Check if the directory already contains data
if [ "$(ls -A "$current_pwd/mounted-nas-do-not-delete-data" 2>/dev/null)" ]; then
    echo "NAS already mounted at $current_pwd/mounted-nas-do-not-delete-data (directory is not empty)"
    exit 0
fi

# Mount NAS server using username and password from .env
sudo mount -t cifs -o username="$NAS_USERNAME",password="$NAS_PASSWORD" "$NAS_PATH" "$current_pwd/mounted-nas-do-not-delete-data"

if [ $? -eq 0 ]; then
    echo "NAS mounted successfully at $current_pwd/mounted-nas-do-not-delete-data"
else
    echo "Failed to mount NAS"
    exit 1
fi
