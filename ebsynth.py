import cv2
import numpy as np
from ._ebsynth import run, EBSYNTH_BACKEND_CPU, EBSYNTH_BACKEND_CUDA, EBSYNTH_BACKEND_AUTO

class ebsynth:
    """
    EBSynth class provides a wrapper around the ebsynth style transfer method.

    Usage:
        ebsynth = EBSynth(style='style.png', guides=[('source1.png', 'target1.png')])
        result_img = ebsynth.run()
    """
    
    def __init__(self, style, guides=[], weight=None, uniformity=3500.0, 
                 patchsize=5, pyramidlevels=6, searchvoteiters=12, 
                 patchmatchiters=6, extrapass3x3=False, backend='cuda'):
        # Handling the style image
        if isinstance(style, str):
            self.style = cv2.imread(style)
        elif isinstance(style, np.ndarray):
            self.style = style
        else:
            raise ValueError("style should be either a file path or a numpy array.")

        # Handling the guide images
        self.guides = []
        for source, target in guides:
            if isinstance(source, str):
                source_img = cv2.imread(source)
            elif isinstance(source, np.ndarray):
                source_img = source
            else:
                raise ValueError("source should be either a file path or a numpy array.")

            if isinstance(target, str):
                target_img = cv2.imread(target)
            elif isinstance(target, np.ndarray):
                target_img = target
            else:
                raise ValueError("target should be either a file path or a numpy array.")

            self.guides.append((source_img, target_img))

        """
        Initialize the EBSynth wrapper.
        
        :param style: path to the style image.
        :param guides: list of tuples containing source and target guide images.
        :param weight: weights for each guide pair. Defaults to 1.0 for each pair.
        :param uniformity: uniformity weight for the style transfer.
        :param patchsize: size of the patches.
        :param pyramidlevels: number of pyramid levels.
        :param searchvoteiters: number of search/vote iterations.
        :param patchmatchiters: number of Patch-Match iterations.
        :param extrapass3x3: whether to perform an extra pass with 3x3 patches.
        :param backend: backend to use ('cpu', 'cuda', or 'auto').
        """
        
        # Store the arguments
        self.style = style
        self.guides = guides
        self.weight = weight if weight else [1.0 for _ in range(len(guides))]
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3
        
        # Define backend constants
        self.backends = {
            'cpu': EBSYNTH_BACKEND_CPU,
            'cuda': EBSYNTH_BACKEND_CUDA,
            'auto': EBSYNTH_BACKEND_AUTO
        }
        self.backend = self.backends[backend]

    def add_guide(self, source, target):
        """
        Add a new guide pair.
        
        :param source: path to the source guide image.
        :param target: path to the target guide image.
        """
        self.guides.append((source, target))
        self.weight.append(1.0)  # Default weight

    def run(self):
        """
        Run the style transfer and return the result image.
        
        :return: styled image as a numpy array.
        """
        if isinstance(self.style, np.ndarray):
            img_style = self.style
        else:
            img_style = cv2.imread(self.style)

        # Prepare the guides
        guides_processed = []
        for idx, (source, target) in enumerate(self.guides):
            if isinstance(source, np.ndarray):
                source_img = source
            else:
                source_img = cv2.imread(source)
            if isinstance(target, np.ndarray):
                target_img = target
            else:
                target_img = cv2.imread(target)
            guides_processed.append((source_img, target_img, self.weight[idx]))

        # Call the run function with the provided arguments
        result = run(img_style, guides_processed, 
                     patch_size=self.patchsize,
                     num_pyramid_levels=self.pyramidlevels,
                     num_search_vote_iters=self.searchvoteiters,
                     num_patch_match_iters=self.patchmatchiters,
                     uniformity_weight=self.uniformity,
                     extraPass3x3=self.extrapass3x3
                    )
        return result
