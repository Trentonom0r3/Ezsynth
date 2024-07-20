import os

import cv2
import numpy as np
import torch
# import torch.nn.functional as F

import tqdm

from .core.raft import RAFT
from .core.utils.utils import InputPadder
from .warp import Warp


class OpticalFlowProcessor:
    valid_flow_methods = [
        "RAFT",
        #   "DeepFlow"
    ]
    valid_model_names = ["sintel", "kitti", "small"]

    def __init__(self, model_name="Sintel", flow_method="RAFT"):
        self.flow_method = flow_method
        self.model_name = model_name
        self.optical_flow = []

    def compute_flow(self, imgsequence):
        if self.flow_method == "RAFT":
            return self._compute_raft_flow(imgsequence)
        # elif self.flow_method == "DeepFlow":
        #     return self._compute_deepflow(imgsequence)
        raise ValueError("Only RAFT is implemented")

    def _compute_raft_flow(self, imgsequence):
        self.flow = RAFT_flow(imgsequence[0], self.model_name)
        self.optical_flow = self.flow.compute_flow(imgsequence)
        return self.optical_flow


class RAFT_flow(Warp):
    device = "cuda"

    def __init__(self, img, model_name="Sintel"):
        """

        Parameters
        ----------
        model_name : str, optional
            DESCRIPTION. The default is 'Sintel'.

        Example
        -------
            flow = RAFT_flow()
            flow.compute_flow(img1, img2)
        For an imgsequence, use the compute_optical_flow method.

            flow = RAFT_flow()
            flow.compute_optical_flow(imgsequence)
        """
        super().__init__(img)
        model_name = "raft-" + model_name + ".pth"
        self.model = torch.nn.DataParallel(
            RAFT(args=self._instantiate_raft_model(model_name))
        )
        model_path = os.path.join(os.path.dirname(__file__), "models", model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"[ERROR] Model file '{model_path}' not found.")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def _instantiate_raft_model(self, model_name):
        from argparse import Namespace

        args = Namespace()
        args.model = model_name
        args.small = False
        args.mixed_precision = False
        return args

    def _load_tensor_from_numpy(self, np_array):
        try:
            tensor = (
                torch.tensor(np_array, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            return tensor
        except Exception as e:
            print(f"[ERROR] Exception in load_tensor_from_numpy: {e}")
            raise e

    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray):
        original_size = img1.shape[1::-1]
        with torch.no_grad():
            img1_tensor = self._load_tensor_from_numpy(img1)
            img2_tensor = self._load_tensor_from_numpy(img2)
            padder = InputPadder(img1_tensor.shape)
            images = padder.pad(img1_tensor, img2_tensor)
            _, flow_up = self.model(images[0], images[1], iters=20, test_mode=True)
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
            cv2.resize(flow_np, original_size)
            return flow_np

    def compute_flow(self, img_frs_seq: list[np.ndarray]):
        optical_flow = []
        zip_img_frs = list(zip(img_frs_seq[:-1], img_frs_seq[1:]))
        for img1, img2 in tqdm.tqdm(
            zip_img_frs, total=len(zip_img_frs), desc="Calculating Flow: "
        ):
            optical_flow.append(self._compute_flow(img1, img2))
        self.optical_flow = optical_flow
        return self.optical_flow
