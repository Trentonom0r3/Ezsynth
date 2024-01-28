import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2
import numpy as np

from . import ebsynth
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


def _process(a: Config, sequences: List[Sequence], guides: Guides):
    style_images_fwd = []
    style_images_bwd = []
    err_fwd = []
    err_bwd = []

    with ThreadPoolExecutor(max_workers = 2) as executor:
        futures = []
        for seq in sequences:
            style_start = next(x.image for x in a.style_frames if x.index == seq.start_frame)
            style_end = next(x.image for x in a.style_frames if x.index == seq.end_frame)

            if style_start is not None and style_end is not None:
                # noinspection PyTypeChecker
                futures.append(("fwd", executor.submit(_run_sequences, guides, seq, 1)))
                # noinspection PyTypeChecker
                futures.append(("bwd", executor.submit(_run_sequences, guides, seq, -1)))

            elif style_start is not None and style_end is None:
                images, _ = _run_sequences(a, guides, seq, 1)
                return [x for x in images if x is not None]

            elif style_start is None and style_end is not None:
                images, _ = _run_sequences(a, guides, seq, -1)
                return [x for x in images if x is not None]

            else:
                raise ValueError("Invalid sequence.")

    for direction, future in futures:
        with threading.Lock():
            try:
                if direction == "fwd":
                    print("Forward")
                    img, err = future.result()
                    if img:
                        style_images_fwd.append(img)
                    if err:
                        err_fwd.append(err)

                else:
                    print("Backward")
                    img, err = future.result()
                    if img:
                        style_images_bwd.append(img)
                    if err:
                        err_bwd.append(err)
            except Exception as e:
                print(f"Process error {e}")

    style_images_b = [img for img in style_images_bwd if img is not None]
    style_images_f = [img for img in style_images_fwd if img is not None]

    sty_fwd = [x for sublist in style_images_f for x in sublist]
    sty_bwd = [x for sublist in style_images_b for x in sublist]
    err_fwd = [x for sublist in err_fwd for x in sublist]
    err_bwd = [x for sublist in err_bwd for x in sublist]

    sty_bwd = sty_bwd[::-1]
    err_bwd = err_bwd[::-1]

    blend_instance = Blend(
        style_fwd = sty_fwd,
        style_bwd = sty_bwd,
        err_fwd = err_fwd,
        err_bwd = err_bwd,
        flow_fwd = guides.flow_fwd,
    )

    final_blends = blend_instance()
    final_blends = [blends for blends in final_blends if blends is not None]

    return final_blends


def _run_sequences(
        a: Config,
        guides: Guides,
        seq: Sequence,
        direction: int,
):
    with threading.Lock():
        stylized_frames = []
        err_list = []

        if direction == 1:
            start_frame, step, style, init, final = (seq.final, -1, seq.style_end, seq.endFrame, seq.begFrame)
        else:
            start_frame, step, style, init, final = (seq.init, 1, seq.style_start, seq.begFrame, seq.endFrame)

        eb = Ebsynth()
        warp = Warp(imgseq[start_frame])

        for i in range(init, final, step):
            config = ebsynth.Config(
                style_image = style,
                guides = [
                    (edge[start_frame], edge[i], 1.0),
                    (imgseq[start_frame], imgseq[i], 6.0),
                ]
            )

            # Commented out section: additional guide and warping
            if i != start_frame:
                eb.add_guide(positional[start_frame - 1] if direction else positional[start_frame], positional[i], 2.0)

                stylized_img = stylized_frames[-1] / 255.0  # Assuming stylized_frames[-1] is already in BGR format

                # Changed from run_warping_from_np to run_warping
                warped_img = warp.run_warping(stylized_img, flow[i] if direction else flow[i - 1])

                warped_img = cv2.resize(warped_img, imgseq[0].shape[1::-1])

                eb.add_guide(style, warped_img, 0.5)

            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)

        print(f"Final Length, Reverse = {direction}: {len(stylized_frames)}. Error Length: {len(err_list)}")
        return stylized_frames, err_list
