import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Union

import cv2
import numpy as np

from . import ebsynth
from .ebsynth import Ebsynth
from .visynth_utils.blend.blender import Blend
from .visynth_utils.config import Config
from .visynth_utils.flow_utils.warp import Warp
from .visynth_utils.guides import Guides, create_guides
# noinspection PyUnresolvedReferences
from .visynth_utils.image_sequence import image_sequence_from_directory
from .visynth_utils.sequences import Sequence, config_to_sequences


class Visynth:
    def __init__(self):
        pass

    def __call__(self, a: Config) -> List[np.ndarray]:
        if len(a.frames) == 0:
            raise ValueError("At least one video frame must be specified.")
        if len(a.style_frames) == 0:
            raise ValueError("At least one style video frame must be specified.")

        sequences = config_to_sequences(a)

        guides = create_guides(a)

        return _process(a, sequences, guides)


def _process(a: Config, sequences: List[Sequence], guides: Guides) -> List[np.ndarray]:
    style_images_fwd = []
    style_images_bwd = []
    err_fwd = []
    err_bwd = []

    with ThreadPoolExecutor(max_workers = 2) as executor:
        futures = []
        for seq in sequences:
            style_start = next((x.image for x in a.style_frames if x.index == seq.start_frame), None)
            style_end = next((x.image for x in a.style_frames if x.index == seq.end_frame), None)

            if style_start is not None and style_end is not None:
                print("Running forward & backward " + str(seq.start_frame) + " <-> " + str(seq.end_frame) + ".")
                # noinspection PyTypeChecker
                futures.append(("fwd", executor.submit(_run_sequences, guides, seq, (style_start, style_end), 1)))
                # noinspection PyTypeChecker
                futures.append(("bwd", executor.submit(_run_sequences, guides, seq, (style_start, style_end), -1)))

            elif style_start is not None and style_end is None:
                print("Running forward " + str(seq.start_frame) + " -> " + str(seq.end_frame) + ".")
                images, _ = _run_sequences(a, guides, seq, (style_start, style_end), 1)
                return [x for x in images if x is not None]

            elif style_start is None and style_end is not None:
                print("Running backward " + str(seq.start_frame) + " <- " + str(seq.end_frame) + ".")
                images, _ = _run_sequences(a, guides, seq, (style_start, style_end), -1)
                return [x for x in images if x is not None]

            else:
                raise ValueError("Cannot find style frame number " + str(seq.start_frame) + " or " + str(seq.end_frame) + ".")

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
        style_frame: Tuple[Union[None, np.ndarray], Union[None, np.ndarray]],
        direction: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    with threading.Lock():
        frames = []
        errors = []

        if direction == 1:
            start_frame = seq.start_frame
            end_frame = seq.end_frame
            step = 1
            style = style_frame[0]
            flow = guides.flow_fwd
            positional = guides.positional_fwd
        else:
            start_frame = seq.end_frame
            end_frame = seq.start_frame
            step = -1
            style = style_frame[1]
            flow = guides.flow_rev
            positional = guides.positional_rev

        eb = Ebsynth()

        warp = Warp(a.frames[start_frame])

        for i in range(start_frame, end_frame, step):
            print("Frame " + str(i) + ".")

            ebsynth_guides = [
                (
                    guides.edge[start_frame],
                    guides.edge[i],
                    1.0,
                ),
                (
                    a.frames[start_frame],
                    a.frames[i],
                    6.0,
                ),
            ]

            if i != start_frame:
                ebsynth_guides.append(
                    (
                        positional[start_frame] if direction == 1 else positional[start_frame - 1],
                        positional[i],
                        2.0,
                    )
                )

                # Assuming frames[-1] is already in BGR format
                frame = frames[-1] / 255.0

                warped_img = warp.run_warping(frame, flow[i - 1] if direction == 1 else flow[i])
                warped_img = cv2.resize(warped_img, a.frames[0].shape[1::-1])

                ebsynth_guides.append(
                    (
                        style,
                        warped_img,
                        0.5,
                    )
                )

            config = ebsynth.Config(
                style_image = style,
                guides = ebsynth_guides,
            )
            frame, err = eb(config)
            frames.append(frame)
            errors.append(err)

        return frames, errors
