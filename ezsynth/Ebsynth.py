import cv2

from ezsynth.utils.EbsynthRunner import *


class Ebsynth:

    def __init__(
            self,
            style,
            guides = None,
            uniformity = 3500.0,
            patch_size = 5,
            pyramid_levels = 6,
            search_voteiters = 12,
            patch_matchiters = 6,
            extra_pass3x3 = True,
            backend = 'auto'
    ):
        """
        Initialize the EBSynth wrapper.      
        :param style: path to the style image, or a numpy array.
        :param guides: list of tuples containing source and target guide images, as file paths or as numpy arrays.
        :param weight: weights for each guide pair. Defaults to 1.0 for each pair.
        :param uniformity: uniformity weight for the style transfer. Defaults to 3500.0.
        :param patch_size: size of the patches. Must be an odd number. Defaults to 5. [5x5 patches]
        :param pyramid_levels: number of pyramid levels. Larger Values useful for things like color transfer. Defaults to 6.
        :param search_voteiters: number of search/vote iterations. Defaults to 12.
        :param patch_matchiters: number of Patch-Match iterations. Defaults to 6.
        :param extra_pass3x3: whether to perform an extra pass with 3x3 patches. Defaults to False.
        :param backend: backend to use ('cpu', 'cuda', or 'auto'). Defaults to 'auto'.
        """
        # self.lock = threading.Lock()
        # Handling the style image
        if isinstance(style, (np.ndarray)):
            self.style = style
        elif isinstance(style, (str)):
            self.style = cv2.imread(style)
        elif style is None:
            print("[INFO] No Style Image Provided. Remember to add a style image to the run() method.")
        else:
            print(type(style))
            raise ValueError("style should be either a file path or a numpy array.")

        # Handling the guide images
        self.guides = []
        # self.eb = LoadDLL()
        self.runner = EbsynthRunner()
        # Store the arguments
        self.style = style
        self.guides = guides
        self.uniformity = uniformity
        self.patchsize = patch_size
        self.pyramidlevels = pyramid_levels
        self.searchvoteiters = search_voteiters
        self.patchmatchiters = patch_matchiters
        self.extrapass3x3 = extra_pass3x3

        # Define backend constants
        self.backends = {
            'cpu': EbsynthRunner.BACKEND_CPU,
            'cuda': EbsynthRunner.BACKEND_CUDA,
            'auto': EbsynthRunner.BACKEND_AUTO
        }
        self.backend = self.backends[backend]

    def clear_guide(self):
        """
        Clear all the guides.
        """
        self.guides = []

    def add_guide(
            self,
            source: Union[str, np.ndarray],
            target: Union[str, np.ndarray],
            weight: Union[int, float, None] = None
    ):
        """
        Add a new guide pair.
        
        :param source: Path to the source guide image or a numpy array.
        :param target: Path to the target guide image or a numpy array.
        :param weight: Weight for the guide pair. Defaults to 1.0.
        """
        if not isinstance(source, (str, np.ndarray)):
            raise ValueError("source should be either a file path or a numpy array.")
        if not isinstance(target, (str, np.ndarray)):
            raise ValueError("target should be either a file path or a numpy array.")
        if not isinstance(weight, (float, int)):
            raise ValueError("weight should be a float or an integer.")

        weight = weight if weight is not None else 1.0
        self.guides.append((source, target, weight))

    def run(self):
        """
        Run the style transfer and return the result image.
        
        :return: styled image as a numpy array.
        """
        # with self.lock:
        if isinstance(self.style, np.ndarray):
            style = self.style
        else:
            style = cv2.imread(self.style)

        # Prepare the guides
        guides_processed = []
        for idx, (source, target, weight) in enumerate(self.guides):
            if isinstance(source, np.ndarray):
                source_img = source
            else:
                source_img = cv2.imread(source)
            if isinstance(target, np.ndarray):
                target_img = target
            else:
                target_img = cv2.imread(target)
            guides_processed.append((source_img, target_img, weight))

        # Call the run function with the provided arguments
        img, err = self.runner.run(style, guides_processed,
                                   patch_size = self.patchsize,
                                   num_pyramid_levels = self.pyramidlevels,
                                   num_search_vote_iters = self.searchvoteiters,
                                   num_patch_match_iters = self.patchmatchiters,
                                   uniformity_weight = self.uniformity,
                                   extra_pass3x3 = self.extrapass3x3
                                   )

        return img, err


def _validate_image(a: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(a, str):
        b = cv2.imread(a)
        if b is None:
            raise ValueError("Cannot read image: " + str(a))
        return b

    elif isinstance(a, np.ndarray):
        if a.shape[-1] != 3:
            raise ValueError("Image a 3-channel numpy array.")
        return a

    else:
        raise ValueError("Image must valid file path or a 3-channel numpy array.")


def _validate_weight(a: Union[int, float, None]) -> float:
    if isinstance(a, int):
        return float(a)

    elif isinstance(a, float):
        return a

    elif a is None:
        return 1.0

    else:
        raise ValueError("Weight should be int or float or none.")
