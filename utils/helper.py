import os

def _grab_all_images(root_path: str) -> list:
    image_paths = []
    image_extensions = ('.jpg', '.jpeg', '.png')
    for directory, _, filenames in os.walk(root_path):
        for filename in filenames:
            if any(filename.lower().endswith(image_extensions)):
                image_paths.append(os.path.join(directory, filename))
    return image_paths