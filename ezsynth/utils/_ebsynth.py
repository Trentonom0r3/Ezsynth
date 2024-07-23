import numpy as np

from ._eb import EbsynthRunner


class ebsynth:
    """
    EBSynth class provides a wrapper around the ebsynth style transfer method.

    Usage:
        ebsynth = ebsynth.ebsynth(style='style.png', guides=[('source1.png', 'target1.png'), 1.0])
        result_img = ebsynth.run()
    """

    def __init__(
        self,
        uniformity=3500.0,
        patchsize=5,
        pyramidlevels=6,
        searchvoteiters=12,
        patchmatchiters=6,
        extrapass3x3=True,
        backend="auto",
    ):
        """
        Initialize the EBSynth wrapper.
        :param style: path to the style image, or a numpy array.
        :param guides: list of tuples containing source and target guide images, as file paths or as numpy arrays.
        :param weight: weights for each guide pair. Defaults to 1.0 for each pair.
        :param uniformity: uniformity weight for the style transfer. Defaults to 3500.0.
        :param patchsize: size of the patches. Must be an odd number. Defaults to 5. [5x5 patches]
        :param pyramidlevels: number of pyramid levels. Larger Values useful for things like color transfer. Defaults to 6.
        :param searchvoteiters: number of search/vote iterations. Defaults to 12.
        :param patchmatchiters: number of Patch-Match iterations. Defaults to 6.
        :param extrapass3x3: whether to perform an extra pass with 3x3 patches. Defaults to False.
        :param backend: backend to use ('cpu', 'cuda', or 'auto'). Defaults to 'auto'.
        """

        self.runner = EbsynthRunner()
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3

        # Define backend constants
        self.backends = {
            "cpu": EbsynthRunner.EBSYNTH_BACKEND_CPU,
            "cuda": EbsynthRunner.EBSYNTH_BACKEND_CUDA,
            "auto": EbsynthRunner.EBSYNTH_BACKEND_AUTO,
        }
        self.backend = self.backends[backend]

    def run(self, style: np.ndarray, guides: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        # Call the run function with the provided arguments
        img, err = self.runner.run(
            style,
            guides,
            patch_size=self.patchsize,
            num_pyramid_levels=self.pyramidlevels,
            num_search_vote_iters=self.searchvoteiters,
            num_patch_match_iters=self.patchmatchiters,
            uniformity_weight=self.uniformity,
            extraPass3x3=self.extrapass3x3,
        )

        return img, err
