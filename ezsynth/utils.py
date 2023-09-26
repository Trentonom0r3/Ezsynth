from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

from .reconstruction import poisson_fusion
from .histogram_blend import HistogramBlender

def _create_selection_mask(err_forward, err_backward):
    selection_masks = []

    for forward, backward in zip(err_forward, err_backward):
        # Check if the images have the same shape
        if forward.shape != backward.shape:
            raise ValueError("Both images must have the same shape.")

        # Create a binary mask where the forward error metric is less than the backward error metric
        selection_mask = np.where( forward < backward, 0, 1)
        selection_mask = selection_mask.astype(np.uint8)

        # Add to the list of masks
        selection_masks.append(selection_mask)

    return selection_masks

# Sample chunk function for histogram preserving blending
def _histogram_preserving_blending_chunk(chunk):
    blended = []
    for forward, backward, mask in chunk:
        if forward.shape[:2] != backward.shape[:2]:
            raise ValueError("Both images must have the same shape.")
        blender = HistogramBlender()
        histogram_blend = blender.blend(forward, backward, mask)
        blended.append(histogram_blend)
    return blended

# Full function for histogram preserving blending
def _histogram_preserving_blending(style_forward, style_backward, selection_masks):
    num_threads = os.cpu_count()
    if num_threads > len(style_forward) or len(style_forward) == 0:
        num_threads = max(1, len(style_forward))
    chunk_size = len(style_forward) // num_threads
    if chunk_size == 0:
        chunk_size = 1

    chunks = [
        list(zip(style_forward[i:i+chunk_size], style_backward[i:i+chunk_size], selection_masks[i:i+chunk_size]))
        for i in range(0, len(style_forward), chunk_size)
    ]

    blended_result = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_histogram_preserving_blending_chunk, chunk) for chunk in chunks]
        for future in futures:
            blended_chunk = future.result()
            blended_result.extend(blended_chunk)

    return blended_result

# Sample chunk function for poisson fusion
def _poisson_fusion_chunk(chunk):
    blended = []
    for hpblended, forward, backward, mask in chunk:
        if forward.shape[:2] != backward.shape[:2]:
            raise ValueError("Both images must have the same shape.")
        poisson_blend = poisson_fusion(hpblended, forward, backward, mask)
        blended.append(poisson_blend)
    return blended

def _poisson_fusion(hpblended, style_forward, style_backward, selection_masks):
    num_threads = os.cpu_count()
    if num_threads > len(hpblended) or len(hpblended) == 0:
        num_threads = max(1, len(hpblended))
    chunk_size = len(hpblended) // num_threads
    if chunk_size == 0:
        chunk_size = 1

    chunks = [
        list(zip(hpblended[i:i+chunk_size], style_forward[i:i+chunk_size], style_backward[i:i+chunk_size], selection_masks[i:i+chunk_size]))
        for i in range(0, len(hpblended), chunk_size)
    ]

    blended_result = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_poisson_fusion_chunk, chunk) for chunk in chunks]
        for future in futures:
            blended_chunk = future.result()
            blended_result.extend(blended_chunk)
    
    return blended_result

def final_blend(style_fwd, style_bwd, selection_masks):
    hpblended = _histogram_preserving_blending(style_fwd, style_bwd, selection_masks)
                    
    final_blends = _poisson_fusion(hpblended, style_fwd, style_bwd, selection_masks)
    
    return final_blends