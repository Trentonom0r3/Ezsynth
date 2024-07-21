import cv2
import numpy as np
import tqdm

from ._ebsynth import ebsynth
from .flow_utils.warp import Warp
from .sequences import EasySequence


def run_first_pass(sequences: list[EasySequence]):
    pass


def run_sequence(
    seq: EasySequence,
    img_frs_seq: list[np.ndarray],
    style_frs: list[np.ndarray],
    edge,
    flow_fwd,
    flow_bwd,
    pos_fwd,
    pos_bwd,
    edg_wgt=1.0,
    img_wgt=6.0,
    pos_wgt=2.0,
    wrp_wgt=0.5,
    uniformity=5000.0,
    patchsize=11,
    pyramidlevels=6,
    searchvoteiters=12,
    patchmatchiters=6,
    extrapass3x3=True,
):
    stylized_frames: list[np.ndarray] = []
    err_list: list[np.ndarray] = []
    ORIGINAL_SIZE = img_frs_seq[0].shape[1::-1]
    if seq.mode == EasySequence.MODE_FWD:
        style = style_frs[seq.style_idxs[0]]
        print("Forward mode")  # Only one style frame
        start = seq.fr_start_idx
        end = seq.fr_end_idx
        step = 1
        stylized_frames.append(style)
        eb = ebsynth(
            style,
            uniformity=uniformity,
            patchsize=patchsize,
            pyramidlevels=pyramidlevels,
            searchvoteiters=searchvoteiters,
            patchmatchiters=patchmatchiters,
            extrapass3x3=extrapass3x3,
        )
        warp = Warp(img_frs_seq[start])
        print(start, end, step)
        for i in tqdm.tqdm(range(start, end, step), "Generating: "):
            eb.add_guide(edge[start], edge[i + 1], edg_wgt)
            eb.add_guide(img_frs_seq[start], img_frs_seq[i + 1], img_wgt)
            if i != start:
                eb.add_guide(pos_fwd[start], pos_fwd[i - 1], pos_wgt)

                # Take the last stylized frame (BGR format)
                stylized_img = stylized_frames[-1] / 255.0

                warped_img = warp.run_warping(stylized_img, flow_fwd[i - 1])
                warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

                eb.add_guide(style, warped_img, wrp_wgt)
            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)
            eb.clear_guide()
        return stylized_frames, err_list
    if seq.mode == EasySequence.MODE_REV:
        style = style_frs[seq.style_idxs[0]]
        print("Reverse mode")  # Only one style frame
        start = seq.fr_end_idx
        end = seq.fr_start_idx
        step = -1
        stylized_frames.append(style)
        eb = ebsynth(style)
        warp = Warp(img_frs_seq[start])
        print(start, end, step)
        for i in tqdm.tqdm(range(start, end, step), desc="Generating: "):
            # Source: Last frame (Style)
            # Target: Previous frame
            eb.add_guide(edge[start], edge[i - 1], edg_wgt)
            eb.add_guide(img_frs_seq[start], img_frs_seq[i - 1], img_wgt)
            if i != start:
                # Get flow from [last to now]
                eb.add_guide(pos_bwd[start - 1], pos_bwd[i], pos_wgt)

                # Take the last stylized frame (BGR format)
                stylized_img = stylized_frames[-1] / 255.0

                warped_img = warp.run_warping(stylized_img, flow_bwd[i])
                warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

                eb.add_guide(style, warped_img, wrp_wgt)
            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)
            eb.clear_guide()
        stylized_frames = stylized_frames[::-1]
        err_list = err_list[::-1]
        return stylized_frames, err_list
    pass
