#!/bin/bash

# Get the current directory path
current_pwd=$(pwd)

sudo umount $current_pwd/mounted-nas-do-not-delete-data

if [ $? -eq 0 ]; then
    echo "NAS successfully unmounted!."
else
    echo "Failed to umount NAS"
fi
