import os


def _grab_all_images(root_path: str) -> list | str:
    image_paths = []
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    try:
        for directory, _, filenames in os.walk(root_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(directory, filename))
    except Exception as E:
        return f"An error occurred: {E}"
    return image_paths


print(_grab_all_images(root_path="mounted-nas-do-not-delete-data"))
