import os

import cv2
import numpy as np
import torch
import tqdm

from .core.utils.utils import InputPadder

class RAFT_flow:
    def __init__(self, model_name="sintel", arch="RAFT"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.arch = arch
        
        if self.arch == "RAFT":
            from .core.raft import RAFT
            model_name = f"raft-{model_name}.pth"
            model_path = os.path.join(os.path.dirname(__file__), "models", model_name)

            if not os.path.exists(model_path):
                raise ValueError(f"[ERROR] Model file '{model_path}' not found.")

            self.model = torch.nn.DataParallel(
                RAFT(args=self._instantiate_raft_model(model_name))
            )
        
        elif self.arch == "EF_RAFT":
            from .core.ef_raft import EF_RAFT
            model_name = f"{model_name}.pth"
            model_path = os.path.join(
                os.path.dirname(__file__), "ef_raft_models", model_name
            )
            if not os.path.exists(model_path):
                raise ValueError(f"[ERROR] Model file '{model_path}' not found.")
            self.model = torch.nn.DataParallel(
                EF_RAFT(args=self._instantiate_raft_model(model_name))
            )
        
        elif self.arch == "FLOW_DIFF":
            try:
                from .flow_diff.flow_diffusion import FlowDiffuser
            except ImportError as e:
                raise ImportError(f"Could not import FlowDiffuser. {e}")
            model_name = "FlowDiffuser-things.pth"
            model_path = os.path.join(os.path.dirname(__file__), "flow_diffusion_models", model_name)
            if not os.path.exists(model_path):
                raise ValueError(f"[ERROR] Model file '{model_path}' not found.")
            self.model = torch.nn.DataParallel(
                FlowDiffuser(args=self._instantiate_raft_model(model_name))
            )
        

        state_dict = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(state_dict)

        self.model.to(self.DEVICE)
        self.model.eval()

    def _instantiate_raft_model(self, model_name):
        from argparse import Namespace

        args = Namespace()
        args.model = model_name
        args.small = False
        args.mixed_precision = False
        return args

    def _load_tensor_from_numpy(self, np_array: np.ndarray):
        try:
            tensor = (
                torch.tensor(np_array, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.DEVICE)
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
            # flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_np = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            cv2.resize(flow_np, original_size)
            return flow_np

    def compute_flow(self, img_frs_seq: list[np.ndarray]):
        optical_flow = []
        total_flows = len(img_frs_seq) - 1
        for i in tqdm.tqdm(range(total_flows), desc="Calculating Flow: "):
            optical_flow.append(self._compute_flow(img_frs_seq[i], img_frs_seq[i + 1]))
        self.optical_flow = optical_flow
        return self.optical_flow
