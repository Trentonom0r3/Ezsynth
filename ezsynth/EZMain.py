import os
import cv2
import numpy as np

from .utils.ezutils import Setup, Runner
from .utils._ebsynth import ebsynth


class Ezsynth:
    """
    edge_method: ["PAGE", "PST", "Classic"] # edge detection
    flow_method: ["RAFT", "DeepFlow"]       # optical flow computation. DeepFlow doesn't work yet
    model: ["sintel", "kitti", "chairs"]    # optical flow
    """

    def __init__(
        self,
        styles,
        imgsequence,
        edge_method="PAGE",
        flow_method="RAFT",
        model="sintel",
        output_folder=None,
    ):
        self.setup = Setup(styles, imgsequence, edge_method, flow_method, model)
        self.output_folder = output_folder
        self.results = None

    def run(self):
        runner = Runner(self.setup)
        self.results = runner.run()
        if self.output_folder is not None:
            self.save()
        return self.results


    def save(self, base_name="output", extension=".png"):
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
            Saver.save_results(
                self.output_folder,
                f"{base_name}{i:03}{extension}",
                self.results[i],
            )
        else:
            print("All results saved successfully")
        return


class Imagesynth:
    INVALID_STYLE_IMG = "style_img must be a str leading to a valid file path or a 3-channel numpy array"

    def __init__(
        self,
        style_img: str | np.ndarray,
        guides: list[str] | list[np.ndarray] = [],
        uniformity=3500.0,
        patchsize=5,
        pyramidlevels=6,
        searchvoteiters=12,
        patchmatchiters=6,
        extrapass3x3=True,
        backend="cuda",
    ):
        """
        Initialize the ebsynth object.

        Parameters
        ----------
        style_img: str or numpy array
            str leading to file path, or numpy array

        guides: tuple of lists
            [[guide 1, guide 2, weight], [guide 1, guide 2, weight], ...]
            guide 1: str leading to file path, or numpy array
            guide 2: str leading to file path, or numpy array
            weight: float

        Example
        -------
        from ezsynth import imagesynth

            STYLE_PATH = "Style.jpg" or np.array
            SOURCE_IMG = "Source.jpg" or np.array
            TARGET_IMG = "Target.jpg" or np.array
            OUTPUT_PATH = "Output.jpg" or None

            eb = imagesynth(style_img = STYLE_PATH)
            eb.add_guide(source = SOURCE_IMG, target = TARGET_IMG, weight = 1.0)
            eb.run(output_path = OUTPUT_PATH)
            or to do something else result = eb.run()

        """
        self.style_img = Validator.validate_image(style_img)
        self.device = "cuda"
        self.eb = ebsynth(
            style=style_img,
            guides=[],
            uniformity=uniformity,
            patchsize=patchsize,
            pyramidlevels=pyramidlevels,
            searchvoteiters=searchvoteiters,
            patchmatchiters=patchmatchiters,
            extrapass3x3=extrapass3x3,
            backend=backend,
        )

    def add_guide(
        self, source: str | np.ndarray, target: str | np.ndarray, weight: float
    ):
        guide = [source, target, weight]
        self.eb.add_guide(guide)

    def clear_guides(self):
        self.eb.clear_guide()

    def run(self, output_path: str | None = None):
        result, _ = self.eb.run()

        if output_path:
            cv2.imwrite(output_path, result)

        return result


class Validator:
    @staticmethod
    def validate_image(img: str | np.ndarray) -> np.ndarray:
        if isinstance(img, str):
            img = cv2.imread(img)
            if img:
                return img
            raise ValueError("Path does not exist")

        if isinstance(img, np.ndarray):
            if img.shape[-1] == 3:
                return img
            raise ValueError(f"Expected 3 channels image. Style shape is {img.shape}")


class Saver:
    @staticmethod
    def save_results(
        output_folder: str, base_file_name: str, result_array: np.ndarray
    ) -> str:
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, base_file_name)
        cv2.imwrite(output_file_path, result_array)
        return output_file_path
