
from typing import List, Union
import os
import re
import cv2
import numpy as np

from utils.sequences import SequenceManager
from utils.guides.guides import GuideFactory
from utils.eb import ebsynth
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
        Preprocessor.__init__(self, style_keys, imgseq)
        GuideFactory.__init__(self, self.imgsequence, edge_method, flow_method, model_name)
        manager = SequenceManager(self.begFrame, self.endFrame, self.styles, self.style_indexes, self.imgindexes)
        self.imgseq = self.imgsequence
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
        self._process()
        
    def _process(self):
        for seq in self.subsequences:
            if seq.style_start is not None and seq.style_end is not None:
                imgs, err_list = self._run_sequence(seq)
                for i in range(len(imgs)):
                    cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/output{i}.jpg", imgs[i])
                img_bwd, err_list_bwd = self._run_sequence(seq, reverse = True)
                for i in range(len(img_bwd)):
                    cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/output_bwd{i}.jpg", img_bwd[i])
               # print(f"Forward: {len(imgs)} | Backward: {len(img_bwd)}")
                # apply blending on the overlapping frames
            elif seq.style_start is not None and seq.style_end is None:
                self._run_sequence(seq)
            elif seq.style_start is None and seq.style_end is not None:
                self._run_sequence(seq, reverse = True)
            else:
                raise ValueError("Invalid sequence.")
            
    def _run_sequence(self, seq, reverse = False):
        stylized_frames = []
        err_list = []
        edge = self.edge_maps
        if reverse:
            edge = edge
            start = seq.final - 1
            end = seq.init
            flow = self.flow_bwd
            positional = self.positional_bwd
            step = -1
            style = seq.style_end
            init = seq.endFrame
            end = seq.begFrame
        else:
            start = seq.init 
            end = seq.final
            flow = self.flow_fwd
            positional = self.positional_fwd
            step = 1
            style = seq.style_start
            init = seq.begFrame
            end = seq.endFrame
           
        eb = ebsynth(style)
        warp = Warp(self.imgsequence[start])
        for i in range(init, end, step):
            print(f"Processing frame {i} of {seq.endFrame}")
            eb.clear_guide()
            eb.add_guide(edge[start], edge[i], 0.5)
            eb.add_guide(self.imgsequence[start], self.imgsequence[i], 6.0)
            
            if i != seq.begFrame and i != seq.endFrame:
                eb.add_guide(positional[start], positional[i], 2.0)
                stylized_img = stylized_frames[-1]
                stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB) / 255.0
                warped_img = warp.run_warping_from_np(stylized_img, flow[i])
                eb.add_guide(style, warped_img, 0.5)
                
            stylized_img, err = eb.run()
            stylized_frames.append(stylized_img)
            err_list.append(err)
            
        return stylized_frames, err_list
    
    def run(self):
        return self._process()
    
                
            
        
        
        
        

    
        
        