import os
import re

import cv2
import numpy as np
import torch
import tqdm


def validate_option(option, values, default):
    return option if option in values else default


def save_to_folder(
    output_folder: str, base_file_name: str, result_array: np.ndarray
) -> str:
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, base_file_name)
    cv2.imwrite(output_file_path, result_array)
    return output_file_path


def validate_and_read_img(img: str | np.ndarray) -> np.ndarray:
    if isinstance(img, str):
        if os.path.isfile(img):
            img = cv2.imread(img)
            return img
        raise ValueError(f"Path does not exist: {img}")

    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3:
            return img
        raise ValueError(f"Expected 3 channels image. Style shape is {img.shape}")


def load_guide(src_path: str, tgt_path: str, weight=1.0):
    src_img = validate_and_read_img(src_path)
    tgt_img = validate_and_read_img(tgt_path)
    return (src_img, tgt_img, weight)


def read_frames_from_paths(lst: list[str]) -> list[np.ndarray]:
    img_arr_seq: list[np.ndarray] = []
    err_frame = -1
    try:
        total = len(lst)
        for err_frame, img_path in tqdm.tqdm(
            enumerate(lst), desc="Reading images: ", total=total
        ):
            img_arr = validate_and_read_img(img_path)
            img_arr_seq.append(img_arr)
        else:
            print(f"Read {len(img_arr_seq)} frames successfully")
            return img_arr_seq
    except Exception as e:
        raise ValueError(f"Error reading frame {err_frame}: {e}")


def read_masks_from_paths(lst: list[str]) -> list[np.ndarray]:
    msk_arr_seq: list[np.ndarray] = []
    err_frame = -1
    try:
        total = len(lst)
        for err_frame, img_path in tqdm.tqdm(
            enumerate(lst), desc="Reading masks: ", total=total
        ):
            msk = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            msk_arr_seq.append(msk)
        else:
            print(f"Read {len(msk_arr_seq)} frames successfully")
            return msk_arr_seq
    except Exception as e:
        raise ValueError(f"Error reading mask frame {err_frame}: {e}")


img_extensions = (".png", ".jpg", ".jpeg")
img_path_pattern = re.compile(r"(\d+)(?=\.(jpg|jpeg|png)$)")


def get_sequence_indices(seq_folder_path: str) -> list[str]:
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
    return sorted(int(img_path_pattern.findall(img_name)[-1][0]) for img_name in lst)


def is_valid_file_path(input_path: str | list[str]) -> bool:
    return isinstance(input_path, str) and os.path.isfile(input_path)


def validate_file_or_folder_to_lst(
    input_paths: str | list[str], type_name=""
) -> list[str]:
    if is_valid_file_path(input_paths):
        return [input_paths]  # type: ignore
    if isinstance(input_paths, list):
        valid_paths = [path for path in input_paths if is_valid_file_path(path)]
        if valid_paths:
            print(f"Received {len(valid_paths)} {type_name} files")
            return valid_paths
    raise FileNotFoundError(f"No valid {type_name} file(s) were found. {input_paths}")


def setup_src_from_folder(
    seq_folder_path: str,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    img_file_paths = get_sequence_indices(seq_folder_path)
    img_idxes = extract_indices(img_file_paths)
    img_frs_seq = read_frames_from_paths(img_file_paths)
    return img_file_paths, img_idxes, img_frs_seq


def setup_masks_from_folder(mask_folder_path: str):
    msk_file_paths = get_sequence_indices(mask_folder_path)
    msk_idxes = extract_indices(msk_file_paths)
    msk_frs_seq = read_masks_from_paths(msk_file_paths)
    return msk_file_paths, msk_idxes, msk_frs_seq


def setup_src_from_lst(paths: list[str], type_name=""):
    val_paths = validate_file_or_folder_to_lst(paths, type_name)
    val_idxes = extract_indices(val_paths)
    frs_seq = read_frames_from_paths(val_paths)
    return val_paths, val_idxes, frs_seq


def save_seq(results: list, output_folder, base_name="output", extension=".png"):
    if not results:
        print("Error: No results to save.")
        return
    for i in range(len(results)):
        save_to_folder(
            output_folder,
            f"{base_name}{i:03}{extension}",
            results[i],
        )
    else:
        print("All results saved successfully")
    return


def replace_zeros_tensor(image: torch.Tensor, replace_value: int = 1) -> torch.Tensor:
    zero_mask = image == 0
    replace_tensor = torch.full_like(image, replace_value)
    return torch.where(zero_mask, replace_tensor, image)


def replace_zeros_np(image: np.ndarray, replace_value: int = 1) -> np.ndarray:
    zero_mask = image == 0
    replace_array = np.full_like(image, replace_value)
    return np.where(zero_mask, replace_array, image)
