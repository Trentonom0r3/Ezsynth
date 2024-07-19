import os
import re

import cv2
import numpy as np


def save_results(
    output_folder: str, base_file_name: str, result_array: np.ndarray
) -> str:
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, base_file_name)
    cv2.imwrite(output_file_path, result_array)
    return output_file_path


def validate_image(img: str | np.ndarray) -> np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img)
        if img:
            return img
        raise ValueError(f"Path does not exist: {img}")

    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3:
            return img
        raise ValueError(f"Expected 3 channels image. Style shape is {img.shape}")


def read_frames_from_paths(lst: list[str]) -> list[np.ndarray]:
    img_arr_seq: list[np.ndarray] = []
    err_frame = -1
    try:
        for err_frame, img_path in enumerate(lst):
            img_arr = validate_image(img_path)
            img_arr_seq.append(img_arr)
        else:
            print(f"Read {len(img_arr_seq)} frames successfully")
            return img_arr_seq
    except Exception as e:
        raise ValueError(f"Error reading frame {err_frame}: {e}")


img_extensions = (".png", ".jpg", ".jpeg")
img_path_pattern = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)")

def get_sequence_indices(self, seq_folder_path: str) -> list[str]:
    if not os.path.isdir(seq_folder_path):
        raise ValueError(f"Path does not exist: {seq_folder_path}")
    file_names = os.listdir(seq_folder_path)
    file_names = sorted(
        file_names,
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", x)],
    )
    img_file_paths = [
        os.path.join(seq_folder_path, file_name)
        for file_name in file_names
        if file_name.lower().endswith(img_extensions)
    ]
    if not img_file_paths:
        raise ValueError("No image files found in the directory.")
    return img_file_paths

def extract_indices(lst: list[str]):
    return sorted(
        int(img_path_pattern.findall(img_name)[-1][0]) for img_name in lst
    )