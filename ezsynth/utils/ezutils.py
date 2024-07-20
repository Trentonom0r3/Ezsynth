# import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tqdm

from ezsynth.aux_utils import (
    extract_indices,
    get_sequence_indices,
    read_frames_from_paths,
    validate_file_or_folder_to_lst,
)

from ._ebsynth import ebsynth
from .blend.blender import Blend
from .flow_utils.warp import Warp
from .guides.guides import GuideFactory
from .sequences import Sequence, SequenceManager

"""
HELPER CLASSES CONTAINED WITHIN THIS FILE:

    - ebsynth
        - __init__
        - add_guide
        - clear_guide
        - __call__
        - run
        - Used to run the underlying Ebsynth pipeline.
        
TODO NEW:
        
TODO REFACTOR:

    - ImageSynth
        - __init__
        - synthesize
        - Used to synthesize a single image. 
        - This is a wrapper around the underlying .pyd file.
        - Optimize, if possible. 
        - Will use the Runner Class, as will the Ezsynth class.
"""


def setup_src_from_folder(
    seq_folder_path: str,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    img_file_paths = get_sequence_indices(seq_folder_path)
    img_idxes = extract_indices(img_file_paths)
    img_frs_seq = read_frames_from_paths(img_file_paths)
    return img_file_paths, img_idxes, img_frs_seq


class Setup:
    def __init__(
        self,
        style_paths: str | list[str],
        seq_folder_path: str,
        edge_method="PAGE",
        flow_method="RAFT",
        model_name="sintel",
        process=False,
    ):
        self.img_file_paths, self.img_idxes, self.img_frs_seq = setup_src_from_folder(
            seq_folder_path
        )
        self.style_paths = validate_file_or_folder_to_lst(style_paths, "style")

        self.begin_fr_idx = self.img_idxes[0]
        self.end_fr_idx = self.img_idxes[-1]

        self.style_idxes = extract_indices(self.style_paths)
        self.num_style_frs = len(self.style_paths)

        if process:
            st = time.time()
            self.guide_factory = GuideFactory(
                img_frs_seq=self.img_frs_seq,
                img_file_paths=self.img_file_paths,
                edge_method=edge_method,
                flow_method=flow_method,
                model_name=model_name,
            )
            self.guides = self.guide_factory.create_all_guides()
            print(f"Guiding took {time.time() - st:.4f} s")
            manager = SequenceManager(
                self.begin_fr_idx,
                self.end_fr_idx,
                self.style_paths,
                self.style_idxes,
                self.img_idxes,
            )
            # self.subsequences = manager._set_sequence()
            self.sequences = manager._set_sequence()
            self.subsequences = ["HAHA"]
            # self.chunk_size = 10
            # self.overlap_frs = 1
            # self.subsequences = [
            #     SequenceManager.generate_subsequences(
            #         sequence, self.chunk_size, self.overlap_frs
            #     )
            #     for sequence in self.sequences
            # ]

    def __str__(self) -> str:
        return (
            f"Setup: Init: {self.begin_fr_idx} - {self.end_fr_idx} | "
            f"Styles: {self.style_idxes} | Subsequences: {[str(sub) for sub in self.subsequences]}"
        )

    def _get_styles(self, style_paths: str | list[str]) -> list[str]:
        """Get the styles either as a list or single string."""
        if isinstance(style_paths, str):
            return [style_paths]
        elif isinstance(style_paths, list):
            return style_paths
        else:
            raise ValueError("Styles must be either a string or a list of strings.")

    def process_sequence(self):
        return process(
            # subseqs=self.subsequences,
            subseqs=self.sequences,
            img_frs_seq=self.img_frs_seq,
            edge_maps=self.guides["edge"],
            flow_fwd=self.guides["flow_fwd"],
            flow_bwd=self.guides["flow_rev"],
            pos_fwd=self.guides["positional_fwd"],
            pos_bwd=self.guides["positional_rev"],
        )


def process(
    subseqs: list[Sequence],
    img_frs_seq,
    edge_maps,
    flow_fwd,
    flow_bwd,
    pos_fwd,
    pos_bwd,
):
    """
    Process sub-sequences using multiprocessing.

    Parameters:
    - subseq: List of sub-sequences to process.
    - imgseq: The sequence of images.
    - edge_maps: The edge maps.
    - flow_fwd: Forward optical flow.
    - flow_bwd: Backward optical flow.
    - pos_fwd: Forward position.
    - pos_bwd: Backward position.

    Returns:
    - imgs: List of processed images.
    """
    # Initialize empty lists to store results
    fwd_styles = []
    bwd_styles = []
    err_fwds = []
    err_bwds = []

    params = {"img_frs_seq": img_frs_seq, "edge": edge_maps}

    for seq in subseqs:
        params["seq"] = seq
        if seq.style_start_fr is not None and seq.style_end_fr is None:
            params["flow"] = flow_fwd
            params["pos"] = pos_fwd
            params["reverse"] = False

            fwd_imgs, _ = run_sequences(**params)
            return fwd_imgs
        if seq.style_start_fr is None and seq.style_end_fr is not None:
            params["flow"] = flow_bwd
            params["pos"] = pos_bwd
            params["reverse"] = True
            bwd_imgs, _ = run_sequences(**params)

            return bwd_imgs
        if seq.style_start_fr is not None and seq.style_end_fr is not None:
            params["flow"] = flow_fwd
            params["pos"] = pos_fwd
            params["reverse"] = False
            print("Running forward pass")
            fwd_pass_imgs, err_fwd = run_sequences(**params)
            fwd_styles.extend(fwd_pass_imgs)
            err_fwds.extend(err_fwd)
            
            params["flow"] = flow_bwd
            params["pos"] = pos_bwd
            params["reverse"] = True
            print("Running backward pass")
            bwd_pass_imgs, err_bwd = run_sequences(**params)
            bwd_styles.extend(bwd_pass_imgs)
            err_bwds.extend(err_bwd)
    
    print("Blend mode awaits. Please don't save yet")
    return fwd_styles, bwd_styles, err_fwds, err_bwds, flow_fwd
            

def run_blending(fwd_styles, bwd_styles, err_fwds, err_bwds, flow_fwd):
    blend_instance = Blend(
        style_fwd=fwd_styles,
        style_bwd=bwd_styles[::-1],
        err_fwd=err_fwds,
        err_bwd=err_bwds[::-1],
        flow_fwd=flow_fwd,
    )
    final_blends = blend_instance()
    return final_blends

def run_sequences(
    img_frs_seq: list[np.ndarray], edge, flow, pos, seq: Sequence, reverse=False
):
    """
    Run the sequence for ebsynth based on the provided parameters.
    Parameters:
        # [Description of each parameter]
    Returns:
        stylized_frames: List of stylized images.
        err_list: List of errors.
    """
    with threading.Lock():
        stylized_frames = []
        err_list = []
        # Initialize variables based on the 'reverse' flag.
        if reverse:
            start, step, style, init, final = (
                seq.final_idx,
                -1,
                seq.style_end_fr,
                seq.end_fr_idx,
                seq.begin_fr_idx,
            )
        else:
            start, step, style, init, final = (
                seq.init_idx,
                1,
                seq.style_start_fr,
                seq.begin_fr_idx,
                seq.end_fr_idx,
            )

        eb = ebsynth(style, guides=[])
        warp = Warp(img_frs_seq[start])
        ORIGINAL_SIZE = img_frs_seq[0].shape[1::-1]
        # Loop through frames.
        stylized_frames.append(eb.style)
        print(start, step, init, final)
        for i in tqdm.tqdm(range(init, final, step), desc="Generating: "):
            eb.add_guide(edge[start], edge[i - 1] if reverse else edge[i + 1], 1.0)
            eb.add_guide(
                img_frs_seq[start],
                img_frs_seq[i - 1] if reverse else img_frs_seq[i + 1],
                6.0,
            )

            # Commented out section: additional guide and warping
            if i != start:
                eb.add_guide(
                    pos[start - 1] if reverse else pos[start],
                    pos[i] if reverse else pos[i - 1],
                    2.0,
                )

                stylized_img = (
                    stylized_frames[-1] / 255.0
                )  # Assuming stylized_frames[-1] is already in BGR format

                warped_img = warp.run_warping(
                    stylized_img, flow[i] if reverse else flow[i - 1]
                )  # Changed from run_warping_from_np to run_warping

                warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

                eb.add_guide(style, warped_img, 0.5)

            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)
            eb.clear_guide()

        print(
            f"Final Length, Reverse = {reverse}: {len(stylized_frames)}. Error Length: {len(err_list)}"
        )
        return stylized_frames, err_list
