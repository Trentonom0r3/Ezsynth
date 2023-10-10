import os
import re
import time
import cv2
import logging
from typing import List, Optional, Union

from .guide_classes import Guides
from .stylizer import Stylizer

# Initialize logging
logging.basicConfig(level=logging.INFO)


class Ezsynth:
    """
    Specialized subclass of ebsynth for video stylization.
    Provides methods to process sequences of images.

    
    """

    def __init__(self, styles: Union[str, List[str]], imgsequence: str,
                 flow_method: str = 'RAFT', edge_method: str = 'PAGE', flow_model: str = 'sintel'):
        """
        Initialize the Ezsynth instance.
        
        Parameters
        ----------
        >>> styles : Union[str, List[str]]
            Path to style image(s).
            (In the form of Style1.jpg, Style2.jpg, Style01.png, Style02.png etc.)
            >>> 3-Channel, 8-bit RGB images only.
            
        >>> imgsequence : str
            Folder Path to image sequence. 
            (In the form of 0001.png, 0002.png, image01.jpg, image02.jpg, etc.)
            >>> 3-Channel, 8-bit RGB images only.
            
        >>> flow_method : str, optional
            Optical flow method, by default 'RAFT'
            >>> options: 'RAFT', 'DeepFlow'
            
        >>> edge_method : str, optional
            Edge method, by default 'PAGE'
            >>> options: 'PAGE', 'PST', 'Classic'
            
        >>> flow_model : str, optional
            Optical flow model, by default 'sintel'
            >>> options: 'sintel', 'kitti'

        Example
        -------
        >>> from ezsynth import Ezsynth
        
        >>> STYLE_PATHS = ["Style1.jpg", "Style2.jpg"]
        >>> IMAGE_FOLDER = "C:/Input"
        >>> OUTPUT_FOLDER = "C:/Output"
        
        >>> ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER)
        >>> ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
        >>> or to do something else results = ez.set_guides().stylize()
        
        Example (For custom Processing Options)
        ---------------------------------------
        >>> from ezsynth import Ezsynth
        
        >>> STYLE_PATHS = ["Style1.jpg", "Style2.jpg"]
        >>> IMAGE_FOLDER = "Input"
        >>> OUTPUT_FOLDER = "Output"
        
        >>> ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER, flow_method='DeepFlow',
                        edge_method='PST')
    
        >>> ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
        >>> or to do something else, results = ez.set_guides().stylize()
        """
        logging.info("Initializing Ezsynth...")
        self.DEVICE = 'cuda'
        self._validate_flow_method(flow_method)
        self._validate_edge_method(edge_method)
        self._validate_model_name(flow_model)
        
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
      
    def _validate_model_name(self, flow_model: str):
        """Validate the model name."""
        valid_models = ['sintel', 'kitti', 'things', 'chairs']
        if flow_model not in valid_models:
            raise ValueError(
                f"Invalid model name. Available options are {valid_models}")
        self.flow_model = flow_model  

    def _validate_flow_method(self, flow_method: str):
        """Validate the flow method."""
        valid_methods = ['RAFT', 'DeepFlow']
        if flow_method not in valid_methods:
            raise ValueError(
                f"Invalid flow method. Available options are {valid_methods}")
        self.flow_method = flow_method

    def _validate_edge_method(self, edge_method: str):
        """Validate the edge method."""
        valid_methods = ['PAGE', 'PST', 'Classic']
        if edge_method not in valid_methods:
            raise ValueError(
                f"Invalid edge method. Available options are {valid_methods}")
        self.edge_method = edge_method

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
            raise ValueError(
                "Styles must be either a string or a list of strings.")

    def _extract_indexes(self, lst: List[str]) -> List[int]:
        """Extract the indexes from the image filenames."""
        pattern = re.compile(r'(\d+)(?=\.(jpg|jpeg|png)$)')
        indexes = [int(pattern.findall(img)[-1][0]) for img in lst]
        return sorted(indexes)

    def set_guides(self) -> None:
        """
        Set the guides for the image sequence initialized with the Ezsynth class.
        
        Accesible Parameters
        --------------------
        >>> flow_guides : List
            Optical flow guides.
        >>> edge_guides : List
            Edge guides.
        >>> g_pos_guides : List
            Dense Correspondence guides.
            
        
        """
        logging.info("Setting guides...")
        self.flow_guides = Guides.compute_optical_flow(
            self, self.imgsequence, self.flow_method, self.flow_model)
        
        self.edge_guides = Guides.compute_edge_guides(
            self, self.imgsequence, self.edge_method)

        g_pos_guides = Guides.create_g_pos(
            self, self.flow_guides, self.imgsequence)

        self.g_pos_guides = g_pos_guides

        self.g_pos_guides_rev = Guides.create_g_pos(
            self, self.flow_guides, self.imgsequence, reverse=True)
        logging.info("Guides set.")
        return self

    def stylize(self, output_path: Optional[str] = None) -> Optional[List]:
        """
        Stylize an image sequence initialized with the Ezsynth class.

        Parameters
        ----------
        output_path : Optional[str], optional
            Path to save the stylized images, by default None

        Returns
        -------
        [list]
            List of stylized images.
        """
        try:
            logging.info("Stylizing...")
            start = time.time()
            stylizer = Stylizer(self.imgsequence, self.styles, self.style_indexes,
                                self.flow_guides, self.edge_guides, self.g_pos_guides,
                                self.g_pos_guides_rev, self.DEVICE)

            stylized_imgs = stylizer.run(output_path)  # If not None, writes images while stylization is occuring.

            if output_path:
                for i, img in enumerate(stylized_imgs):
                    cv2.imwrite(f"{output_path}/final_output{i}.png", img)
            end = time.time()
            logging.info(f"Stylization complete. Time taken: {end-start} seconds.")
            return stylized_imgs

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
