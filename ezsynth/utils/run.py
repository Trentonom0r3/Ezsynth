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
    patchsize=7,
    pyramidlevels=6,
    searchvoteiters=12,
    patchmatchiters=6,
    extrapass3x3=True,
):
    stylized_frames: list[np.ndarray] = []
    err_list: list[np.ndarray] = []
    ORIGINAL_SIZE = img_frs_seq[0].shape[1::-1]
    if seq.mode != EasySequence.MODE_BLN:
        # Only one style frame
        style = style_frs[seq.style_idxs[0]]
        eb = ebsynth(
            style,
            uniformity=uniformity,
            patchsize=patchsize,
            pyramidlevels=pyramidlevels,
            searchvoteiters=searchvoteiters,
            patchmatchiters=patchmatchiters,
            extrapass3x3=extrapass3x3,
        )
        stylized_frames.append(style)

        if seq.mode == EasySequence.MODE_FWD:
            print("Forward mode")
            start = seq.fr_start_idx
            end = seq.fr_end_idx
            step = 1
            is_forward = True
        elif seq.mode == EasySequence.MODE_REV:
            print("Reverse mode")
            start = seq.fr_end_idx
            end = seq.fr_start_idx
            step = -1
            is_forward = False
        warp = Warp(img_frs_seq[start])
        print(start, end, step)
        for i in tqdm.tqdm(range(start, end, step), "Generating: "):
            eb.add_guide(edge[start], edge[i + step], edg_wgt)
            eb.add_guide(img_frs_seq[start], img_frs_seq[i + step], img_wgt)

            if i != start:
                eb.add_guide(
                    pos_fwd[start] if is_forward else pos_bwd[start - 1],
                    pos_fwd[i - 1] if is_forward else pos_bwd[i],
                    pos_wgt,
                )
                # Take the last stylized frame (BGR format)
                stylized_img = stylized_frames[-1] / 255.0

                warped_img = warp.run_warping(
                    stylized_img, flow_fwd[i - 1] if is_forward else flow_bwd[i]
                )
                warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

                eb.add_guide(style, warped_img, wrp_wgt)
            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)
            eb.clear_guide()

        if seq.mode == EasySequence.MODE_REV:
            stylized_frames = stylized_frames[::-1]
            err_list = err_list[::-1]
        return stylized_frames, err_list
    pass
