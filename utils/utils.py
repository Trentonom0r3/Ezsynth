
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Union
import os
import re
import cv2
import numpy as np
from utils.blend.blender import Blend

from utils.sequences import SequenceManager
from utils.guides.guides import GuideFactory
from utils.ebsynth import ebsynth
from utils.flow_utils.warp import Warp
"""
HELPER CLASSES CONTAINED WITHIN THIS FILE:

    - Preprocessor
        - _get_image_sequence
        - _get_styles
        - _extract_indexes
        - _read_frames
        -Used to preprocess the image sequence.

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

class Preprocessor:
    def __init__(self, styles: Union[str, List[str]], img_sequence: str):
        """
        Initialize the Preprocessor class.

        Parameters
        ----------
        styles : Union[str, List[str]]
            Style(s) used for the sequence. Can be a string or a list of strings.
        img_sequence : str
            Directory path containing the image sequence.

        Examples
        --------
        >>> preprocessor = Preprocessor("Style1", "/path/to/image_sequence")
        >>> preprocessor.styles
        ['Style1']

        >>> preprocessor = Preprocessor(["Style1", "Style2"], "/path/to/image_sequence")
        >>> preprocessor.styles
        ['Style1', 'Style2']
        """
        self.imgsequence = []
        imgsequence = self._get_image_sequence(img_sequence)
        self.imgseq = imgsequence
        self.imgindexes = self._extract_indexes(imgsequence)
        self._read_frames(imgsequence)
        self.begFrame = self.imgindexes[0]
        self.endFrame = self.imgindexes[-1]
        self.styles = self._get_styles(styles)
        self.style_indexes = self._extract_indexes(self.styles)
        self.num_styles = len(self.styles)

    def _get_image_sequence(self, img_sequence: str) -> List[str]:
        """Get the image sequence from the directory."""
        if not os.path.isdir(img_sequence):
            raise ValueError("img_sequence must be a valid directory.")
        filenames = sorted(os.listdir(img_sequence),
                           key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
        img_files = [os.path.join(img_sequence, fname) for fname in filenames if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            raise ValueError("No image files found in the directory.")
        return img_files

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
 
class Setup(Preprocessor, GuideFactory):
    
    def __init__(self, style_keys, imgseq, edge_method="PAGE",
                 flow_method="RAFT", model_name="sintel"):
        prepro = Preprocessor(style_keys, imgseq)
        GuideFactory.__init__(self, prepro.imgsequence, prepro.imgseq, edge_method, flow_method, model_name)
        manager = SequenceManager(prepro.begFrame, prepro.endFrame, prepro.styles, prepro.style_indexes, prepro.imgindexes)
        self.imgseq = prepro.imgsequence
        self.subsequences = manager._set_sequence()
        self.guides = self.create_all_guides()  # works well, just commented out since it takes a bit to run.
        
    def __call__(self):
        return self.guides, self.subsequences
    
    def __str__(self):
        return f"Setup: Init: {self.begFrame} - {self.endFrame} | Styles: {self.style_indexes} | Subsequences: {[str(sub) for sub in self.subsequences]}"
    
class Runner:
    def __init__(self, setup):
        self.setup = setup
        self.guides, self.subsequences = self.setup()
        self.imgsequence = self.setup.imgseq
        self.flow_fwd = self.guides["flow_fwd"]
        self.flow_bwd = self.guides["flow_rev"]
        self.edge_maps = self.guides["edge"]
        self.positional_fwd = self.guides["positional_fwd"]
        self.positional_bwd = self.guides["positional_rev"]

    def run(self):
        return process(self.subsequences, self.imgsequence, self.edge_maps, self.flow_fwd, self.flow_bwd, self.positional_fwd, self.positional_bwd)

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
    
    with threading.Lock():
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []  # Keep your existing list to store the futures
            for seq in subseq:
                print(f"Submitting sequence: {seq}")
                
                # Your existing logic to submit tasks remains the same
                if seq.style_start is not None and seq.style_end is not None:
                    futures.append(("fwd", executor.submit(run_sequences, seq.style_start, 
                                                        imgseq, edge_maps, flow_fwd,
                                                        flow_bwd, pos_fwd, pos_bwd, seq)))
                    futures.append(("bwd", executor.submit(run_sequences, seq.style_end, 
                                                        imgseq, edge_maps, flow_fwd,
                                                        flow_bwd, pos_fwd, pos_bwd, seq, True)))
                elif seq.style_start is not None and seq.style_end is None:
                    futures.append(("fwd", executor.submit(run_sequences, seq.style_start, 
                                                        imgseq, edge_maps, flow_fwd,
                                                        flow_bwd, pos_fwd, pos_bwd, seq)))
                elif seq.style_start is None and seq.style_end is not None:
                    futures.append(("bwd", executor.submit(run_sequences, seq.style_end,
                                                        imgseq, edge_maps, flow_fwd,
                                                        flow_bwd, pos_fwd, pos_bwd, seq, True)))
                else:
                    raise ValueError("Invalid sequence.")

    with threading.Lock():
        for direction, future in futures:
            try:
                img, err = future.result()
                if direction == "fwd":
                    if img:
                        style_imgs_fwd.extend(img)
                    if err:
                        err_fwd.append(err)
                else:  # direction is "bwd"
                    if img:
                        style_imgs_bwd.extend(img[:: -1])
                    if err:
                        err_bwd.append(err[:: -1])
            except TimeoutError:
                print("TimeoutError")
            except Exception as e:
                print(f"Exception: {e}")
                
    t1 = time.time()
    
    # Initialize the Blend class
    blend_instance = Blend(style_fwd=style_imgs_fwd, 
                        style_bwd=style_imgs_bwd, 
                        err_fwd=err_fwd, 
                        err_bwd=err_bwd, 
                        flow_fwd=flow_fwd)

    # Invoke the __call__ method to perform blending
    final_blends = blend_instance()
    
    t2 = time.time()
    print(f"Time taken to blend: {t2 - t1}")
    
    return final_blends

def run_sequences(style, imgseq, edge_maps, flow_fwd, flow_bwd,
                  pos_fwd, pos_bwd, seq, reverse=False):
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
        edge, start, end, flow, positional, step, style, init, final = (
            edge_maps, seq.final - 1, seq.init, flow_bwd, pos_bwd, -1, seq.style_end, seq.endFrame, seq.begFrame)
    else:
        edge, start, end, flow, positional, step, style, init, final = (
            edge_maps, seq.init, seq.final, flow_fwd, pos_fwd, 1, seq.style_start, seq.begFrame, seq.endFrame)

    eb = ebsynth(style, guides=[])
    warp = Warp(imgseq[start])
    ORIGINAL_SIZE = imgseq[0].shape[1::-1]
    # Loop through frames.
    for i in range(init, final, step):
        eb.clear_guide()
        eb.add_guide(edge[start], edge[i], 0.5)
        eb.add_guide(imgseq[start], imgseq[i], 6.0)

        # Commented out section: additional guide and warping
        if i != seq.begFrame and i != seq.endFrame:
            eb.add_guide(positional[start], positional[i], 2.0)
            stylized_img = stylized_frames[-1] / 255.0  # Assuming stylized_frames[-1] is already in BGR format
            warped_img = warp.run_warping(stylized_img, flow[i] if reverse else flow[i - 1])  # Changed from run_warping_from_np to run_warping
            warped_img = cv2.resize(warped_img, ORIGINAL_SIZE)

            eb.add_guide(style, warped_img, 0.5)
            
        stylized_img, err = eb.run()
        stylized_frames.append(stylized_img)
        err_list.append(err)

    return stylized_frames, err_list


        
    
          
        


                
            
        
        
        
        

    
        
        