import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np

from .ebsynth import Ebsynth
from .visynth_utils.blend.blender import Blend
# noinspection PyUnresolvedReferences
from .visynth_utils.config import Config, image_sequence_from_directory
from .visynth_utils.flow_utils.warp import Warp
from .visynth_utils.guides import Guides, create_guides
from .visynth_utils.sequences import Sequence, config_to_sequences


class Visynth:
    def __init__(self):
        pass

    def __call__(self, a: Config) -> List[tuple[int, np.ndarray]]:
        if len(a.frames) == 0:
            raise ValueError("At least one video frame must be specified.")
        if len(a.style_frames) == 0:
            raise ValueError("At least one style video frame must be specified.")

        sequences = config_to_sequences(a)
        print("Sequences: " + ", ".join(map(lambda x: str(x.start_frame) + "-" + str(x.end_frame), sequences)))

        guides = create_guides(a)

        return _process(a, sequences, guides)


def _process(config: Config, sequences: List[Sequence], guides: Guides):
    style_images_fwd = []
    style_images_bwd = []
    err_fwd = []
    err_bwd = []

    with ThreadPoolExecutor(max_workers = 2) as executor:
        futures = []
        for seq in sequences:
            if seq.style_start is not None and seq.style_end is not None:
                futures.append(("fwd", executor.submit(_run_sequences, imgseq, edge_maps, flow_fwd, pos_fwd, seq)))
                futures.append(("bwd", executor.submit(_run_sequences, imgseq, edge_maps, flow_bwd, pos_bwd, seq, True)))

            elif seq.style_start is not None and seq.style_end is None:
                fwd_img, fwd_err = _run_sequences(imgseq, edge_maps, flow_fwd, pos_fwd, seq)
                fwd_imgs = [img for img in fwd_img if img is not None]
                return fwd_imgs

            elif seq.style_start is None and seq.style_end is not None:
                bwd_img, bwd_err = _run_sequences(imgseq, edge_maps, flow_bwd, pos_bwd, seq, True)
                bwd_imgs = [img for img in bwd_img if img is not None]
                return bwd_imgs

            else:
                raise ValueError("Invalid sequence.")

    for direction, future in futures:
        with threading.Lock():
            try:
                img, err = future.result()
                if direction == "fwd":
                    print("Forward")
                    if img:
                        style_images_fwd.append(img)
                    if err:
                        err_fwd.append(err)
                else:
                    print("Backward")
                    if img:
                        style_images_bwd.append(img)
                    if err:
                        err_bwd.append(err)
            except Exception as e:
                print(f"List Creation Exception: {e}")

    style_images_b = [img for img in style_images_bwd if img is not None]
    style_images_f = [img for img in style_images_fwd if img is not None]

    sty_fwd = [img for sublist in style_images_f for img in sublist]
    sty_bwd = [img for sublist in style_images_b for img in sublist]
    err_fwd = [img for sublist in err_fwd for img in sublist]
    err_bwd = [img for sublist in err_bwd for img in sublist]

    sty_bwd = sty_bwd[::-1]
    err_bwd = err_bwd[::-1]

    blend_instance = Blend(
        style_fwd = sty_fwd,
        style_bwd = sty_bwd,
        err_fwd = err_fwd,
        err_bwd = err_bwd,
        flow_fwd = flow_fwd,
    )

    final_blends = blend_instance()
    final_blends = [blends for blends in final_blends if blends is not None]

    return final_blends


def _run_sequences(
        imgseq,
        edge,
        flow,
        positional,
        seq,
        reverse = False
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
                seq.final, -1, seq.style_end, seq.endFrame, seq.begFrame)
        else:
            start, step, style, init, final = (
                seq.init, 1, seq.style_start, seq.begFrame, seq.endFrame)

        eb = Ebsynth(style, guides = [])
        warp = Warp(imgseq[start])
        ORIGINAL_SIZE = imgseq[0].shape[1::-1]
        # Loop through frames.
        for i in range(init, final, step):
            eb.clear_guide()
            eb.add_guide(edge[start], edge[i], 1.0)
            eb.add_guide(imgseq[start], imgseq[i], 6.0)

            # Commented out section: additional guide and warping
            if i != start:
                eb.add_guide(positional[start - 1] if reverse else positional[start], positional[i], 2.0)

                stylized_img = stylized_frames[-1] / 255.0  # Assuming stylized_frames[-1] is already in BGR format

                warped_img = warp.run_warping(stylized_img, flow[i] if reverse else flow[
                    i - 1])  # Changed from run_warping_from_np to run_warping

                warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

                eb.add_guide(style, warped_img, 0.5)

            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)

        print(f"Final Length, Reverse = {reverse}: {len(stylized_frames)}. Error Length: {len(err_list)}")
        return stylized_frames, err_list
