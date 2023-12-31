from .utils.ezutils import *
from .utils.guides.guides import *

class Ezsynth:
    """
    `ezsynth` is the main class for the ezsynth package.
    
    Parameters
    ----------
    styles : list
        The paths to the style images.
    imgsequence : str
        The path to the folder containing the input images.
    edge_method : str, optional
        The method for edge detection, default is "PAGE".
        Options are "PAGE", "PST", and "Classic".
    flow_method : str, optional
        The method for optical flow computation, default is "RAFT".
        Options are "RAFT" and "DeepFlow".
    model : str, optional
        The model name for optical flow, default is "sintel".
        Options are "sintel", "kitti", and "chairs".
    """
    def __init__(self, styles, imgsequence, edge_method = "PAGE",
                 flow_method = "RAFT", model = "sintel", output_folder = None):
        
        self.setup = Setup(styles, imgsequence, edge_method, flow_method, model)
        self.output_folder = output_folder
        self.results = None

        
    def run(self):
        runner = Runner(self.setup)
        self.results = runner.run()
        if self.output_folder is not None:
            self.save()
        return self.results
    
    @staticmethod
    def save_results(output_folder, base_file_name, result_array):
        """
        Save the numpy array result to the specified directory and return the file path.
        """
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, base_file_name)
        cv2.imwrite(output_file_path, result_array)
        return output_file_path
    
    def save(self, base_name = "output", extension = ".png"):
        """
        Save the results to the specified directory.
        
        If the results are a single image, save it as base_name + extension.
        If the results are a list of images, save them as base_name + 000 + extension, base_name + 001 + extension, etc.
        
        If the results are None, print an error message.
        """
        if self.results is None:
            print("Error: No results to save.")
            return
        for i in range(len(self.results)):
            self.save_results(self.output_folder, base_name + str(i).zfill(3) + extension, self.results[i])
        return

class Imagesynth:
    INVALID_STYLE_IMG = 'style_img must be a str leading to a valid file path or a 3-channel numpy array'
    INVALID_GUIDE_FORMAT = 'guides must be a list of lists in the format [guide 1, guide 2, weight]'
    INVALID_OUTPUT_PATH = 'output_path must be a str leading to a valid file path or None'
    def __init__(self, style_img, guides = [], uniformity = 3500.0, 
                 patchsize = 5, pyramidlevels = 6, searchvoteiters = 12, 
                 patchmatchiters = 6, extrapass3x3 = True, backend = 'cuda'):
        """
        Initialize the ebsynth object.
        
        Parameters
        ----------
        style_img: str or numpy array
        >>> str leading to file path, or numpy array
        
        guides: tuple of lists
        >>> [[guide 1, guide 2, weight], [guide 1, guide 2, weight], ...]
        >>> guide 1: str leading to file path, or numpy array
        >>> guide 2: str leading to file path, or numpy array
        >>> weight: float
        
        Example
        -------
        from ezsynth import imagesynth
        
        >>> STYLE_PATH = "Style.jpg" or np.array
        >>> SOURCE_IMG = "Source.jpg" or np.array
        >>> TARGET_IMG = "Target.jpg" or np.array
        >>> OUTPUT_PATH = "Output.jpg" or None
        
        >>> eb = imagesynth(style_img = STYLE_PATH)
        >>> eb.add_guide(source = SOURCE_IMG, target = TARGET_IMG, weight = 1.0)
        >>> eb.run(output_path = OUTPUT_PATH)
        >>> or to do something else result = eb.run()
        
        """
        self.style_img = self._validate_style_img(style_img)
        self.device = 'cuda'
        self.eb = ebsynth(style = style_img, guides = [], uniformity = uniformity,
                            patchsize = patchsize, pyramidlevels = pyramidlevels,
                            searchvoteiters = searchvoteiters, patchmatchiters = patchmatchiters,
                            extrapass3x3 = extrapass3x3, backend = backend)
    
    def add_guide(self, source, target, weight):
        """
        Add a guide to the ebsynth object.
        
        Parameters
        ----------
        source: str or numpy array
        >>> str leading to file path, or numpy array
        
        target: str or numpy array
        >>> str leading to file path, or numpy array
        
        weight: float
        """
        
        self._validate_guide([source, target, weight])
        self.eb.add_guide(source, target, weight)
        
    def clear_guides(self):
        self.eb.clear_guide()
    
    @staticmethod
    def _validate_image(img):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(Imagesynth.INVALID_STYLE_IMG)
        elif isinstance(img, np.ndarray):
            if img.shape[-1] != 3:
                raise ValueError(Imagesynth.INVALID_STYLE_IMG)
        else:
            raise ValueError(Imagesynth.INVALID_STYLE_IMG)
        return img
    
    def _validate_style_img(self, style_img):
        return self._validate_image(style_img)
    
    def _validate_guide(self, guide):
        if len(guide) != 3:
            raise ValueError(Imagesynth.INVALID_GUIDE_FORMAT)
        self._validate_image(guide[0])
        self._validate_image(guide[1])
        if not isinstance(guide[2], float):
            raise ValueError('weight must be a float')
        
    def _validate_output_path(self, output_path):
        if not (isinstance(output_path, str) or output_path is None):
            raise ValueError(Imagesynth.INVALID_OUTPUT_PATH)
        return output_path

    def run(self, output_path = None):
        """
        Run ebsynth.
        
        Parameters
        ----------
        output_path: str(optional)
        >>> str leading to file path
        
        returns: numpy array
        
        """
        output_path = self._validate_output_path(output_path)
        result, _ = self.eb.run(output_nnf = True)

        if output_path:
            cv2.imwrite(output_path, result)
            
        return result
