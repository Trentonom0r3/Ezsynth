import os
from typing import Literal, List

import cv2
import numpy as np
import torch

from .core.raft import RAFT
from .core.utils.utils import InputPadder
from .warp import Warp


class OpticalFlowProcessor:
    def __init__(
            self,
            method: Literal["RAFT", "DeepFlow"] = "RAFT",
            model: Literal["sintel", "kitti", "chairs"] = "sintel",
            device: torch.device = torch.device("cpu"),
    ):
        self.method = method
        self.model = model
        self.device = device

    def __call__(self, images: List[np.ndarray]):
        if self.method == "RAFT":
            self.flow = RAFT_flow(images[0], self.model)
            return self.flow.__iter__(images)

        elif self.method == "DeepFlow":
            raise NotImplementedError("DeepFlow method is not implemented.")

        else:
            raise ValueError("Unknown optical flow method.")


# noinspection PyPep8Naming
class RAFT_flow(Warp):

    def __init__(self, img, model_name = 'Sintel'):
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
        self.model = torch.nn.DataParallel(RAFT(args = self._instantiate_raft_model(model_name)))
        model_path = os.path.join(os.path.dirname(__file__), 'models', model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"[ERROR] Model file '{model_path}' not found.")

        self.model.load_state_dict(torch.load(model_path, map_location = self.device))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    # noinspection PyMethodMayBeStatic
    def _instantiate_raft_model(self, model_name):
        from argparse import Namespace
        args = Namespace()
        args.model = model_name
        args.small = False
        args.mixed_precision = False
        return args

    def __iter__(self, images):
        for img1, img2 in zip(images[:-1], images[1:]):
            yield self._compute_flow(img1, img2)

    def _compute_flow(self, img1, img2):
        original_size = img1.shape[1::-1]
        with torch.no_grad():
            img1_tensor = self._load_tensor_from_numpy(img1)
            img2_tensor = self._load_tensor_from_numpy(img2)
            padder = InputPadder(img1_tensor.shape)
            images = padder.pad(img1_tensor, img2_tensor)
            _, flow_up = self.model(images[0], images[1], iters = 20, test_mode = True)
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
            cv2.resize(flow_np, original_size)
            return flow_np

    def _load_tensor_from_numpy(self, np_array):
        try:
            return torch.tensor(np_array, dtype = torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Exception in load_tensor_from_numpy: {e}")
            raise e
