import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import cv2
import numpy as np

from ._ebsynth import ebsynth
from .blend.blender import Blend
from .flow_utils.warp import Warp
from .guides.guides import GuideFactory
from .sequences import SequenceManager


def _get_image_paths(path: str) -> List[str]:
    """Get file names of image sequence from the directory."""
    if not os.path.isdir(path):
        raise ValueError("img_sequence must be a valid directory.")
    a = sorted(os.listdir(path))
    return [
        os.path.join(path, filename) for filename in a
        if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


class Preprocessor:
    def __init__(self, styles: Union[str, List[str]], img_sequence: str):
        image_paths = _get_image_paths(img_sequence)
        self.imgsequence = []
        self.imgseq = image_paths
        self.imgindexes = self._extract_indexes(image_paths)
        self._read_frames(image_paths)
        self.begFrame = self.imgindexes[0]
        self.endFrame = self.imgindexes[-1]
        self.styles = self._get_styles(styles)
        self.style_indexes = self._extract_indexes(self.styles)
        self.num_styles = len(self.styles)

    def _get_styles(self, styles: Union[str, List[str]]) -> List[str]:
        """Get the styles either as a list or single string."""
        if isinstance(styles, str):
            return [styles]
        elif isinstance(styles, list):
            return styles
        else:
            raise ValueError("Styles must be either a string or a list of strings.")

    def _extract_indexes(self, lst: List[str]) -> List[int]:
        """Extract the indexes from the image filenames."""
        pattern = re.compile(r'(\d+)(?=\.(jpg|jpeg|png)$)')
        return sorted(int(pattern.findall(img)[-1][0]) for img in lst)

    def _read_frames(self, lst: List[str]) -> List[np.ndarray]:
        """Read the image frames."""
        try:
            for img in lst:
                cv2_img = cv2.imread(img)
                self.imgsequence.append(cv2_img)

            return self.imgsequence
        except Exception as e:
            raise ValueError(f"Error reading image frames: {e}")


def setup(style_keys, input_path = "input", edge_method = "PAGE", flow_method = "RAFT", model_name = "sintel"):
    prepro = Preprocessor(style_keys, input_path)
    guide = GuideFactory(prepro.imgsequence, prepro.imgseq, edge_method, flow_method, model_name)
    manager = SequenceManager(prepro.begFrame, prepro.endFrame, prepro.styles, prepro.style_indexes,
                              prepro.imgindexes)
    subsequences = manager._set_sequence()
    guides = guide.create_all_guides()  # works well, just commented out since it takes a bit to run.
    return prepro.imgsequence, subsequences, guides


def process(subseq, imgseq, edge_maps, flow_fwd, flow_bwd, pos_fwd, pos_bwd):
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

    with ThreadPoolExecutor(max_workers = 2) as executor:
        futures = []  # Keep your existing list to store the futures
        for seq in subseq:
            print(f"Submitting sequence:")
            # Your existing logic to submit tasks remains the same
            if seq.style_start is not None and seq.style_end is not None:

                futures.append(("fwd", executor.submit(run_sequences, imgseq, edge_maps, flow_fwd,
                                                       pos_fwd, seq)))

                futures.append(("bwd", executor.submit(run_sequences, imgseq, edge_maps,
                                                       flow_bwd, pos_bwd, seq, True)))

            elif seq.style_start is not None and seq.style_end is None:

                fwd_img, fwd_err = run_sequences(imgseq, edge_maps, flow_fwd,
                                                 pos_fwd, seq)
                fwd_imgs = [img for img in fwd_img if img is not None]

                return fwd_imgs
            elif seq.style_start is None and seq.style_end is not None:

                bwd_img, bwd_err = run_sequences(imgseq, edge_maps,
                                                 flow_bwd, pos_bwd, seq, True)
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
    blend_instance = Blend(style_fwd = sty_fwd,
                           style_bwd = sty_bwd,
                           err_fwd = err_fwd,
                           err_bwd = err_bwd,
                           flow_fwd = flow_fwd)

    # Invoke the __call__ method to perform blending
    final_blends = blend_instance()
    final_blends = [blends for blends in final_blends if blends is not None]

    # t2 = time.time()
    # print(f"Time taken to blend: {t2 - t1}")

    return final_blends


def run_sequences(imgseq, edge, flow,
                  positional, seq, reverse = False):
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

        eb = ebsynth(style, guides = [])
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
