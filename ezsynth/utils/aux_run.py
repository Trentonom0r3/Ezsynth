# import gc

import cv2
import numpy as np

# import torch
import tqdm

from ezsynth.aux_classes import PositionalGuide
from ezsynth.utils._ebsynth import ebsynth
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow
from ezsynth.utils.flow_utils.warp import Warp
from ezsynth.utils.sequences import EasySequence


class RunConfig:
    def __init__(
        self,
        edg_wgt=1.0,
        img_wgt=6.0,
        pos_wgt=2.0,
        wrp_wgt=0.5,
        uniformity=3500.0,
        patchsize=7,
        pyramidlevels=6,
        searchvoteiters=12,
        patchmatchiters=6,
        extrapass3x3=True,
    ) -> None:
        self.edg_wgt = edg_wgt
        self.img_wgt = img_wgt
        self.pos_wgt = pos_wgt
        self.wrp_wgt = wrp_wgt
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3

    def get_ebsynth_cfg(self):
        return {
            "uniformity": self.uniformity,
            "patchsize": self.patchsize,
            "pyramidlevels": self.pyramidlevels,
            "searchvoteiters": self.searchvoteiters,
            "patchmatchiters": self.patchmatchiters,
            "extrapass3x3": self.extrapass3x3,
        }


def run_scratch(
    seq: EasySequence,
    img_frs_seq: list[np.ndarray],
    style_frs: list[np.ndarray],
    edge,
    cfg: RunConfig,
):
    stylized_frames: list[np.ndarray] = []
    err_list: list[np.ndarray] = []
    ORIGINAL_SIZE = img_frs_seq[0].shape[1::-1]

    style = style_frs[seq.style_idxs[0]]
    stylized_frames.append(style)

    start, end, step, is_forward = (
        get_forward(seq) if seq.mode == EasySequence.MODE_FWD else get_backward(seq)
    )

    warp = Warp(img_frs_seq[start])
    print(start, end, step)
    flows = []
    poses = []

    rafter = RAFT_flow(model_name="sintel")
    pos_guider = PositionalGuide()

    eb = ebsynth(**cfg.get_ebsynth_cfg())
    eb.runner.initialize_libebsynth()

    for i in tqdm.tqdm(range(start, end, step), "Generating"):
        if is_forward:
            flow = rafter._compute_flow(img_frs_seq[i], img_frs_seq[i + step])
        else:
            flow = rafter._compute_flow(img_frs_seq[i + step], img_frs_seq[i])
        flows.append(flow)

        poster = pos_guider.create_from_flow(flow, ORIGINAL_SIZE, warp)
        poses.append(poster)

        stylized_img = stylized_frames[-1] / 255.0
        warped_img = warp.run_warping(stylized_img, flow * (-step))
        warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

        stylized_img, err = eb.run(
            style,
            guides=[
                (edge[start], edge[i + step], cfg.edg_wgt),
                (img_frs_seq[start], img_frs_seq[i + step], cfg.img_wgt),
                (poses[0], poster, cfg.pos_wgt),
                (style, warped_img, cfg.wrp_wgt),
            ],
        )
        stylized_frames.append(stylized_img)
        err_list.append(err)
    if not is_forward:
        stylized_frames = stylized_frames[::-1]
        err_list = err_list[::-1]
        flows = flows[::-1]
        poses = poses[::-1]

    return stylized_frames, err_list, flows, poses


def get_forward(seq: EasySequence):
    start = seq.fr_start_idx
    end = seq.fr_end_idx
    step = 1
    is_forward = True
    return start, end, step, is_forward


def get_backward(seq: EasySequence):
    start = seq.fr_end_idx
    end = seq.fr_start_idx
    step = -1
    is_forward = False
    return start, end, step, is_forward
