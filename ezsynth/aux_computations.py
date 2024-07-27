import tqdm
import numpy as np
from .edge_detection import EdgeDetector


def precompute_edge_guides(
    img_frs_seq: list[np.ndarray], edge_method: str
) -> list[np.ndarray]:
    edge_detector = EdgeDetector(edge_method)
    edge_maps = []
    for img_fr in tqdm.tqdm(img_frs_seq, desc="Calculating edge maps"):
        edge_maps.append(edge_detector.compute_edge(img_fr))
    return edge_maps

