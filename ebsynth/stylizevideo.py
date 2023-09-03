import os
import re
import cv2
import logging
from typing import List, Optional, Union
from utils.guide_classes import Guides
from utils.stylizer import Stylizer

# Initialize logging
logging.basicConfig(level=logging.INFO)

class ezsynth:
    """
    Specialized subclass of ebsynth for video stylization.
    Provides methods to process sequences of images.

    Parameters
    ----------
    styles : Union[str, List[str]]
        Style image file paths, in ascending order.
    imgsequence : str
        Path to folder containing image sequence.
    flow_method : str, optional
        Optical flow method to use, by default 'RAFT'.
    edge_method : str, optional
        Edge detection method to use, by default 'PAGE'.
    DEVICE : str, optional
        Device to use for computation, by default 'cuda'.

    Attributes
    ----------
    flow_method : str
        Optical flow method to use.
    edge_method : str
        Edge detection method to use.
    DEVICE : str
        Device to use for computation.
    flow_guides : None
        Optical flow guides.
    edge_guides : None
        Edge guides.
    g_pos_guides : None
        g_pos guides.
    imgsequence : List[str]
        List of image sequence paths.
    imgindexes : List[int]
        List of image indexes.
    begFrame : int
        Index of the first frame.
    endFrame : int
        Index of the last frame.
    style_indexes : List[int]
        List of style indexes.
    num_styles : int
        Number of style images.
    styles : List[str]
        List of style images.
    """

    def __init__(self, styles: Union[str, List[str]], imgsequence: str,
                 flow_method: str = 'RAFT', edge_method: str = 'PAGE', 
                 DEVICE: str = 'cuda'):
        """
        Initialize the EzSynth instance.
        """
        logging.info("Initializing EzSynth...")
        self._validate_flow_method(flow_method)
        self._validate_edge_method(edge_method)
        self._validate_device(DEVICE)

        self.flow_guides = None
        self.edge_guides = None
        self.g_pos_guides = None

        self.imgsequence = self._get_image_sequence(imgsequence)
        self.imgindexes = self._extract_indexes(self.imgsequence)
        self.begFrame = self.imgindexes[0]
        self.endFrame = self.imgindexes[-1]

        self.styles = self._get_styles(styles)
        self.style_indexes = self._extract_indexes(self.styles)
        self.num_styles = len(self.styles)

    def _validate_flow_method(self, flow_method: str):
        """Validate the flow method."""
        valid_methods = ['RAFT', 'DeepFlow']
        if flow_method not in valid_methods:
            raise ValueError(f"Invalid flow method. Available options are {valid_methods}")
        self.flow_method = flow_method

    def _validate_edge_method(self, edge_method: str):
        """Validate the edge method."""
        valid_methods = ['PAGE', 'PST', 'Classic']
        if edge_method not in valid_methods:
            raise ValueError(f"Invalid edge method. Available options are {valid_methods}")
        self.edge_method = edge_method

    def _validate_device(self, DEVICE: str):
        """Validate the computation device."""
        valid_devices = ['cuda', 'cpu']
        if DEVICE not in valid_devices:
            raise ValueError(f"Invalid device. Available options are {valid_devices}")
        self.DEVICE = DEVICE

    def _get_image_sequence(self, imgsequence: str) -> List[str]:
        """Get the image sequence from the directory."""
        if os.path.isdir(imgsequence):
            filenames = sorted(os.listdir(imgsequence), 
                               key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
            return [os.path.join(imgsequence, fname) for fname in filenames if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            raise ValueError("imgsequence must be a valid directory.")

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
        indexes = [int(pattern.findall(img)[-1][0]) for img in lst]
        return sorted(indexes)

    def _set_guides(self) -> None:
        """Set the guides for flow, edge and g_pos."""
        self.flow_guides = Guides.compute_optical_flow(self, self.imgsequence, self.flow_method)
        self.edge_guides = Guides.compute_edge_guides(self, self.imgsequence, self.edge_method)
        self.g_pos_guides = Guides.create_g_pos(self, self.flow_guides, self.imgsequence)
        self.g_pos_guides_rev = Guides.create_g_pos(self, self.flow_guides, self.imgsequence, reverse=True)
        for i in range(len(self.g_pos_guides)):
            cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/gpos{i}.png", self.g_pos_guides[i])
        return self
    
    def stylize(self, output_path: Optional[str] = None) -> Optional[List]:
        """
        Stylize an image sequence initialized with the EzSynth class.

        Parameters
        ----------
        output_path : Optional[str], optional
            Path to save the stylized images, by default None

        Returns
        -------
        Optional[List]
            List of stylized images.
        """
        try:
            
            stylizer = Stylizer(self.imgsequence, self.styles, self.style_indexes,
                                self.flow_guides, self.edge_guides, self.g_pos_guides, 
                                self.g_pos_guides_rev, self.DEVICE)
            
            stylized_imgs = stylizer.run()
            
            if output_path:
                for i, img in enumerate(stylized_imgs):
                    cv2.imwrite(f"{output_path}/output{i}.png", img)

            return stylized_imgs

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
