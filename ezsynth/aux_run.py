import cv2
import numpy as np
import tqdm

from .aux_classes import PositionalGuide, RunConfig
from .utils._ebsynth import ebsynth
from .utils.blend.blender import Blend
from .utils.flow_utils.OpticalFlow import RAFT_flow
from .utils.flow_utils.warp import Warp
from .sequences import EasySequence


def run_a_pass(
    seq: EasySequence,
    seq_mode: str,
    img_frs_seq: list[np.ndarray],
    style: np.ndarray,
    edge: list[np.ndarray],
    cfg: RunConfig,
    rafter: RAFT_flow,
    eb: ebsynth,
):
    stylized_frames: list[np.ndarray] = [style]
    err_list: list[np.ndarray] = []
    ORIGINAL_SIZE = img_frs_seq[0].shape[1::-1]

    start, end, step, is_forward = (
        get_forward(seq) if seq_mode == EasySequence.MODE_FWD else get_backward(seq)
    )
    warp = Warp(img_frs_seq[start])
    print(f"{'Forward' if is_forward else 'Reverse'} mode. {start=}, {end=}, {step=}")
    flows = []
    poses = []
    pos_guider = PositionalGuide()

    for i in tqdm.tqdm(range(start, end, step), "Generating"):
        flow = get_flow(img_frs_seq, rafter, step, is_forward, i)
        flows.append(flow)

        poster = pos_guider.create_from_flow(flow, ORIGINAL_SIZE, warp)
        poses.append(poster)
        warped_img = get_warped_img(stylized_frames, ORIGINAL_SIZE, step, warp, flow)

        stylized_img, err = eb.run(
            style,
            guides=[
                (edge[start], edge[i + step], cfg.edg_wgt),  # Slower with premask
                (img_frs_seq[start], img_frs_seq[i + step], cfg.img_wgt),
                (poses[0], poster, cfg.pos_wgt),
                (style, warped_img, cfg.wrp_wgt),  # Slower with premask
            ],
        )
        stylized_frames.append(stylized_img)
        err_list.append(err)

    if not is_forward:
        stylized_frames = stylized_frames[::-1]
        err_list = err_list[::-1]
        flows = flows[::-1]

    return stylized_frames, err_list, flows


def get_warped_img(
    stylized_frames: list[np.ndarray], ORIGINAL_SIZE, step: int, warp: Warp, flow
):
    stylized_img = stylized_frames[-1] / 255.0
    warped_img = warp.run_warping(stylized_img, flow * (-step))
    warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)
    return warped_img


def get_flow(
    img_frs_seq: list[np.ndarray],
    rafter: RAFT_flow,
    step: int,
    is_forward: bool,
    i: int,
):
    if is_forward:
        flow = rafter._compute_flow(img_frs_seq[i], img_frs_seq[i + step])
    else:
        flow = rafter._compute_flow(img_frs_seq[i + step], img_frs_seq[i])
    return flow


def run_scratch(
    seq: EasySequence,
    img_frs_seq: list[np.ndarray],
    style_frs: list[np.ndarray],
    edge: list[np.ndarray],
    cfg: RunConfig,
    rafter: RAFT_flow,
    eb: ebsynth,
):
    if seq.mode == EasySequence.MODE_BLN and cfg.only_mode != EasySequence.MODE_NON:
        print(f"{cfg.only_mode} Only")
        stylized_frames, err_list, _ = run_a_pass(
            seq,
            cfg.only_mode,
            img_frs_seq,
            style_frs[seq.style_idxs[0]]
            if cfg.only_mode == EasySequence.MODE_FWD
            else style_frs[seq.style_idxs[1]],
            edge,
            cfg,
            rafter,
            eb,
        )
        return stylized_frames, err_list

    if seq.mode != EasySequence.MODE_BLN:
        stylized_frames, err_list, _ = run_a_pass(
            seq,
            seq.mode,
            img_frs_seq,
            style_frs[seq.style_idxs[0]],
            edge,
            cfg,
            rafter,
            eb,
        )
        return stylized_frames, err_list

    print("Blending mode")

    style_fwd, err_fwd, flow_fwd = run_a_pass(
        seq,
        EasySequence.MODE_FWD,
        img_frs_seq,
        style_frs[seq.style_idxs[0]],
        edge,
        cfg,
        rafter,
        eb,
    )

    style_bwd, err_bwd, _ = run_a_pass(
        seq,
        EasySequence.MODE_REV,
        img_frs_seq,
        style_frs[seq.style_idxs[1]],
        edge,
        cfg,
        rafter,
        eb,
    )

    return run_blend(img_frs_seq, style_fwd, style_bwd, err_fwd, err_bwd, flow_fwd, cfg)


def run_blend(
    img_frs_seq: list[np.ndarray],
    style_fwd: list[np.ndarray],
    style_bwd: list[np.ndarray],
    err_fwd: list[np.ndarray],
    err_bwd: list[np.ndarray],
    flow_fwd: list[np.ndarray],
    cfg: RunConfig,
):
    blender = Blend(**cfg.get_blender_cfg())

    err_masks = blender._create_selection_mask(err_fwd, err_bwd)

    warped_masks = blender._warping_masks(img_frs_seq[0], flow_fwd, err_masks)

    hist_blends = blender._hist_blend(style_fwd, style_bwd, warped_masks)

    blends = blender._reconstruct(style_fwd, style_bwd, warped_masks, hist_blends)

    if not cfg.skip_blend_style_last:
        blends.append(style_bwd[-1])

    return blends, warped_masks


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
