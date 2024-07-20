import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import time

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
    style_imgs_fwd = []
    err_fwd = []
    style_imgs_bwd = []
    err_bwd = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for seq in subseqs:
            print(seq)
            if seq.style_start_fr is not None and seq.style_end_fr is None:
                fwd_img, fwd_err = run_sequences(
                    img_frs_seq, edge_maps, flow_fwd, pos_fwd, seq
                )
                return fwd_img
            # Doesnt work
            if seq.style_start_fr is None and seq.style_end_fr is not None:
                print(f"Run from frame {seq.style_end_fr}")
                bwd_img, bwd_err = run_sequences(
                    img_frs_seq, edge_maps, flow_bwd, pos_bwd, seq, True
                )
                bwd_imgs = [img for img in bwd_img if img is not None]

                return bwd_imgs
            # This case does not work
            if seq.style_start_fr is not None and seq.style_end_fr is not None:
                print(f"{seq.style_start_fr=}")
                print(f"{seq.style_end_fr=}")
                futures.append(
                    (
                        "fwd",
                        executor.submit(
                            run_sequences,
                            img_frs_seq,
                            edge_maps,
                            flow_fwd,
                            pos_fwd,
                            seq,
                        ),
                    )
                )

                futures.append(
                    (
                        "bwd",
                        executor.submit(
                            run_sequences,
                            img_frs_seq,
                            edge_maps,
                            flow_bwd,
                            pos_bwd,
                            seq,
                            True,
                        ),
                    )
                )
            else:
                raise ValueError("Invalid sequence.")

    for direction, future in futures:
        with threading.Lock():
            try:
                img, err = future.result()
                if direction == "fwd":
                    print("Forward")
                    if img:
                        style_imgs_fwd.append(img)
                    if err:
                        err_fwd.append(err)
                else:  # direction is "bwd"
                    print("Backward")
                    if img:
                        style_imgs_bwd.append(img)
                    if err:
                        err_bwd.append(err)
            except TimeoutError:
                print("TimeoutError")
            except Exception as e:
                print(f"List Creation Exception: {e}")
                continue
    # C:\Users\tjerf\Desktop\Testing\src\Testvids\Output
    style_imgs_b = [img for img in style_imgs_bwd if img is not None]
    style_imgs_f = [img for img in style_imgs_fwd if img is not None]

    sty_fwd = [img for sublist in style_imgs_f for img in sublist]
    sty_bwd = [img for sublist in style_imgs_b for img in sublist]
    err_fwd = [img for sublist in err_fwd for img in sublist]
    err_bwd = [img for sublist in err_bwd for img in sublist]

    sty_bwd = sty_bwd[::-1]
    err_bwd = err_bwd[::-1]
    # check length of sty_fwd and sty_bwd
    # if length of one is zero, skip blending and return the other
    # Initialize the Blend class
    blend_instance = Blend(
        style_fwd=sty_fwd,
        style_bwd=sty_bwd,
        err_fwd=err_fwd,
        err_bwd=err_bwd,
        flow_fwd=flow_fwd,
    )

    # Invoke the __call__ method to perform blending
    final_blends = blend_instance()
    final_blends = [blends for blends in final_blends if blends is not None]

    # t2 = time.time()
    # print(f"Time taken to blend: {t2 - t1}")

    return final_blends


def run_sequences(
    imgseq: list[np.ndarray], edge, flow, pos, seq: Sequence, reverse=False
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
        warp = Warp(imgseq[start])
        ORIGINAL_SIZE = imgseq[0].shape[1::-1]
        # Loop through frames.
        for i in range(init, final, step):
            eb.clear_guide()
            eb.add_guide(edge[start], edge[i], 1.0)
            eb.add_guide(imgseq[start], imgseq[i], 6.0)

            # Commented out section: additional guide and warping
            if i != start:
                eb.add_guide(
                    pos[start - 1] if reverse else pos[start],
                    pos[i],
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

        print(
            f"Final Length, Reverse = {reverse}: {len(stylized_frames)}. Error Length: {len(err_list)}"
        )
        return stylized_frames, err_list
