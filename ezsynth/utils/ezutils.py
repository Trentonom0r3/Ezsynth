# import os
# import threading
import time
# from concurrent.futures import ThreadPoolExecutor

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
from .sequences import EasySequence, Sequence, SequenceManager

from .run import run_scratch, run_sequence

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
        cal_flow_bwd=False,
    ):
        self.img_file_paths, self.img_idxes, self.img_frs_seq = setup_src_from_folder(
            seq_folder_path
        )
        self.style_paths = validate_file_or_folder_to_lst(style_paths, "style")

        self.begin_fr_idx = self.img_idxes[0]
        self.end_fr_idx = self.img_idxes[-1]

        self.style_idxes = extract_indices(self.style_paths)
        self.num_style_frs = len(self.style_paths)
        self.style_frs = read_frames_from_paths(self.style_paths)

        manager = SequenceManager(
            self.begin_fr_idx,
            self.end_fr_idx,
            self.style_paths,
            self.style_idxes,
            self.img_idxes,
        )

        self.sequences = manager.create_sequences()
        if process:
            st = time.time()
            self.guide_factory = GuideFactory(
                img_frs_seq=self.img_frs_seq,
                img_file_paths=self.img_file_paths,
                edge_method=edge_method,
                flow_method=flow_method,
                model_name=model_name,
            )
            self.guides = self.guide_factory.create_all_guides(cal_flow_bwd)
            print(f"Guiding took {time.time() - st:.4f} s")

    def process_sequence(self):
        stylized_frames = []
        err_list = []
        flows = []
        poses = []
        forward_skip_last = False
        reverse_remove_last = False
        num_seqs = len(self.sequences)
        is_mid = False
        for i, seq in enumerate(self.sequences):
            if i == 0:
                continue
            # if i == 0 and i < num_seqs - 1:
            #     if seq.mode == EasySequence.MODE_FWD:
            #         forward_skip_last = True
            #     elif seq.mode == EasySequence.MODE_REV:
            #         reverse_remove_last = True
            # if i > 0 and i < num_seqs - 1:
            #     is_mid = True
            tmp_style, tmp_err, tmp_fl, tmp_p = run_scratch(
                seq,
                self.img_frs_seq,
                self.style_frs,
                edge=self.guides["edge"],
                # forward_skip_last = forward_skip_last,
                # is_mid=is_mid
            )
            
            if reverse_remove_last: # Remove the style frame itself
                tmp_style = tmp_style[:-1]
                tmp_err = tmp_err[:-1]
                    
            stylized_frames.extend(tmp_style)
            err_list.extend(tmp_err)
            flows.extend(tmp_fl)
            poses.extend(tmp_p)
            # forward_skip_last = False
            # reverse_remove_last = False
            print(len(stylized_frames))
        
        return stylized_frames, err_list, flows, poses
        # return process(
        #     subseqs=self.sequences,
        #     img_frs_seq=self.img_frs_seq,
        #     edge_maps=self.guides["edge"],
        #     flow_fwd=self.guides["flow_fwd"],
        #     flow_bwd=self.guides["flow_rev"],
        #     pos_fwd=self.guides["positional_fwd"],
        #     pos_bwd=self.guides["positional_rev"],
        #     forward_only=forward_only
        # )
        # stylized_frames, err_list = [], []
        # for seq in self.sequences:
        #     tmp_style, tmp_err = run_sequence(
        #         seq,
        #         self.img_frs_seq,
        #         self.style_frs,
        #         edge=self.guides["edge"],
        #         flow_fwd=self.guides["flow_fwd"],
        #         flow_bwd=self.guides["flow_rev"],
        #         pos_fwd=self.guides["positional_fwd"],
        #         pos_bwd=self.guides["positional_rev"],
        #     )
        #     stylized_frames.extend(tmp_style)
        #     err_list.extend(tmp_err)
        # return stylized_frames, err_list


def process(
    subseqs: list[Sequence],
    img_frs_seq,
    edge_maps,
    flow_fwd,
    flow_bwd,
    pos_fwd,
    pos_bwd,
    forward_only: bool,
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

    no_blend = []

    params = {"img_frs_seq": img_frs_seq, "edge": edge_maps}

    for seq in subseqs:
        params["seq"] = seq
        if seq.style_start_fr is not None and seq.style_end_fr is None:
            params["flow"] = flow_fwd
            params["pos"] = pos_fwd
            params["reverse"] = False

            fwd_pass_imgs, err_fwd = run_sequences(**params)
            if seq.is_all:
                return fwd_pass_imgs
            if seq.is_blend and not forward_only:
                fwd_styles.extend(fwd_pass_imgs)
                err_fwds.extend(err_fwd)
            else:
                no_blend.extend(fwd_pass_imgs)
            continue
        if seq.style_start_fr is None and seq.style_end_fr is not None:
            params["flow"] = flow_bwd
            params["pos"] = pos_bwd
            params["reverse"] = True
            bwd_pass_imgs, err_bwd = run_sequences(**params)
            if seq.is_all:
                return bwd_pass_imgs
            bwd_styles.extend(bwd_pass_imgs)
            err_bwds.extend(err_bwd)
            continue
        if seq.style_start_fr is not None and seq.style_end_fr is not None:
            params["flow"] = flow_fwd
            params["pos"] = pos_fwd
            params["reverse"] = False
            print("Running forward pass")
            fwd_pass_imgs, err_fwd = run_sequences(**params)
            fwd_styles.extend(fwd_pass_imgs)
            err_fwds.extend(err_fwd)
            if forward_only:
                no_blend.extend(fwd_pass_imgs)
                continue

            params["flow"] = flow_bwd
            params["pos"] = pos_bwd
            params["reverse"] = True
            print("Running backward pass")
            bwd_pass_imgs, err_bwd = run_sequences(**params)
            bwd_styles.extend(bwd_pass_imgs)
            err_bwds.extend(err_bwd)
            continue
    if forward_only:
        return no_blend
    print("Blend mode awaits. Please don't save yet")
    return fwd_styles, bwd_styles, err_fwds, err_bwds, flow_fwd, no_blend


def run_blending(
    fwd_styles,
    bwd_styles,
    err_fwds,
    err_bwds,
    flow_fwd,
    use_gpu=False,
    use_lsqr=True,
    use_poisson_cupy=False,
    poisson_maxiter=None,
):
    blend_instance = Blend(
        style_fwd=fwd_styles,
        style_bwd=bwd_styles[::-1],
        err_fwd=err_fwds,
        err_bwd=err_bwds[::-1],
        flow_fwd=flow_fwd,
        use_gpu=use_gpu,
        use_lsqr=use_lsqr,
        use_poisson_cupy=use_poisson_cupy,
        poisson_maxiter=poisson_maxiter,
    )
    final_blends = blend_instance.run_final_blending()
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
