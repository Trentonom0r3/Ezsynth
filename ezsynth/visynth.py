from typing import List, Tuple

import cv2
import numpy as np

from . import ebsynth
from .ebsynth import Ebsynth
from .visynth_utils.config import Config
from .visynth_utils.flow_utils.warp import Warp
from .visynth_utils.guides import Guides, config_to_guides
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

        guides = config_to_guides(a)

        configs = config_and_guides_and_sequences_to_ebsynth_configs_1pass(a, guides, sequences)
        frames = _process(configs)

        return []


def _process(configs: List[Tuple[int, ebsynth.Config]]) -> List[np.ndarray]:
    eb = Ebsynth()
    acc = []
    for i, c in sorted(configs):
        print("Frame " + str(i) + ".")
        acc.append(eb(c)[0])
    return acc


def config_and_guides_and_sequences_to_ebsynth_configs_1pass(
        a: Config,
        guides: Guides,
        sequences: List[Sequence],
) -> List[Tuple[int, ebsynth.Config]]:
    acc: List[Tuple[int, ebsynth.Config]] = []

    for b in sequences:
        style_start = next((x[1] for x in a.style_frames if x[0] == b.start_frame), None)
        style_end = next((x[1] for x in a.style_frames if x[0] == b.end_frame), None)

        if style_start is None and style_end is None:
            raise ValueError("Cannot find style frame number " + str(b.start_frame) + " or " + str(b.end_frame) + ".")

        if style_start is not None:
            print("Running forward " + str(b.start_frame) + " -> " + str(b.end_frame) + ".")
            acc += _to_ebsynth_configs_helper_1pass(a, guides, b, style_start, 1)

        if style_end is not None:
            print("Running backward " + str(b.start_frame) + " <- " + str(b.end_frame) + ".")
            acc += _to_ebsynth_configs_helper_1pass(a, guides, b, style_end, -1)

    return acc


def _to_ebsynth_configs_helper_1pass(
        a: Config,
        guides: Guides,
        sequence: Sequence,
        style_frame: np.ndarray,
        direction: int,
) -> List[Tuple[int, ebsynth.Config]]:
    acc: List[Tuple[int, ebsynth.Config]] = []

    if direction == 1:
        start_frame = sequence.start_frame
        end_frame = sequence.end_frame
        step = 1
        flow = guides.flow_fwd
        positional = guides.positional_fwd
    else:
        start_frame = sequence.end_frame
        end_frame = sequence.start_frame
        step = -1
        flow = guides.flow_rev
        positional = guides.positional_rev

    warp = Warp(a.frames[start_frame])

    for i in range(start_frame, end_frame, step):
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
                    style_frame,
                    warped_img,
                    0.5,
                )
            )

        config = ebsynth.Config(
            style_image = style_frame,
            guides = ebsynth_guides,
        )
        acc.append((i, config))

    return acc


def config_and_guides_and_sequences_to_ebsynth_configs_2pass(
        a: Config,
        guides: Guides,
        sequences: List[Sequence],
        frames: List[np.ndarray],
) -> List[Tuple[int, ebsynth.Config]]:
    acc: List[Tuple[int, ebsynth.Config]] = []

    for b in sequences:
        style_start = next((x[1] for x in a.style_frames if x[0] == b.start_frame), None)
        style_end = next((x[1] for x in a.style_frames if x[0] == b.end_frame), None)

        if style_start is None and style_end is None:
            raise ValueError("Cannot find style frame number " + str(b.start_frame) + " or " + str(b.end_frame) + ".")

        if style_start is not None:
            print("Running forward " + str(b.start_frame) + " -> " + str(b.end_frame) + ".")
            acc += _to_ebsynth_configs_helper_2pass(a, guides, b, style_start, 1, frames)

        if style_end is not None:
            print("Running backward " + str(b.start_frame) + " <- " + str(b.end_frame) + ".")
            acc += _to_ebsynth_configs_helper_2pass(a, guides, b, style_end, -1, frames)

    return acc


def _to_ebsynth_configs_helper_2pass(
        a: Config,
        guides: Guides,
        sequence: Sequence,
        style_frame: np.ndarray,
        direction: int,
        frames: List[np.ndarray],
) -> List[Tuple[int, ebsynth.Config]]:
    acc: List[Tuple[int, ebsynth.Config]] = []

    if direction == 1:
        start_frame = sequence.start_frame
        end_frame = sequence.end_frame
        step = 1
        flow = guides.flow_fwd
        positional = guides.positional_fwd
    else:
        start_frame = sequence.end_frame
        end_frame = sequence.start_frame
        step = -1
        flow = guides.flow_rev
        positional = guides.positional_rev

    warp = Warp(a.frames[start_frame])

    for i in range(start_frame, end_frame, step):
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
                    style_frame,
                    warped_img,
                    0.5,
                )
            )

        config = ebsynth.Config(
            style_image = style_frame,
            guides = ebsynth_guides,
        )
        acc.append((i, config))

    return acc
