import cv2
import numpy as np
import tqdm


def apply_mask(image: np.ndarray, mask: np.ndarray):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image.astype(np.uint8)


def apply_masks(images: list[np.ndarray], masks: list[np.ndarray]):
    len_img = len(images)
    len_msk = len(masks)
    if len_img != len_msk:
        raise ValueError(f"[{len_img=}], [{len_msk=}]")

    masked_images = []
    for i in range(len_img):
        masked_images.append(apply_mask(images[i], masks[i]))

    return masked_images


def apply_masks_idxes(
    images: list[np.ndarray], masks: list[np.ndarray], img_idxes: list[int]
):
    masked_images = []
    for i, idx in enumerate(img_idxes):
        masked_images.append(apply_mask(images[i], masks[idx]))
    return masked_images


def apply_masked_back(
    original: np.ndarray, processed: np.ndarray, mask: np.ndarray, feather_radius=0
):
    if feather_radius > 0:
        mask_blurred = cv2.GaussianBlur(mask, (feather_radius, feather_radius), 0)
        mask_blurred = mask_blurred.astype(np.float32) / 255.0

        mask_inv_blurred = 1.0 - mask_blurred

        # Expand dimensions to match the number of channels in the original image
        mask_blurred_expanded = np.expand_dims(mask_blurred, axis=-1)
        mask_inv_blurred_expanded = np.expand_dims(mask_inv_blurred, axis=-1)

        background = original * mask_inv_blurred_expanded
        foreground = processed * mask_blurred_expanded

        # Combine the background and foreground
        result = background + foreground
        result = result.astype(np.uint8)

    else:
        mask = mask.astype(np.float32) / 255.0
        mask_inv = 1.0 - mask
        mask_expanded = np.expand_dims(mask, axis=-1)
        mask_inv_expanded = np.expand_dims(mask_inv, axis=-1)
        background = original * mask_inv_expanded
        foreground = processed * mask_expanded
        result = background + foreground
        result = result.astype(np.uint8)

    return result


def apply_masked_back_seq(
    img_frs_seq: list[np.ndarray],
    styled_msk_frs: list[np.ndarray],
    mask_frs_seq: list[np.ndarray],
    feather=0,
):
    len_img = len(img_frs_seq)
    len_stl = len(styled_msk_frs)
    len_msk = len(mask_frs_seq)

    if len_img != len_stl != len_msk:
        raise ValueError(f"Lengths not match. [{len_img=}, {len_stl=}, {len_msk=}]")

    backed_seq = []

    for i in tqdm.tqdm(range(len_img), desc="Adding masked back"):
        backed_seq.append(
            apply_masked_back(
                img_frs_seq[i], styled_msk_frs[i], mask_frs_seq[i], feather
            )
        )

    return backed_seq
