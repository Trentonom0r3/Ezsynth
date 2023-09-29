import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core.raft import RAFT
from .core.utils.utils import InputPadder

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = 'cuda'


def instantiate_raft_model(model_name):
    from argparse import Namespace
    args = Namespace()
    args.model = model_name
    args.small = False
    args.mixed_precision = False
    return args


class OpticalFlowProcessor:

    # Class-level dictionary to map descriptive names to model filenames
    MODEL_NAME_OPTIONS = {
        "sintel": "raft-sintel.pth",
        "kitti": "raft-kitti.pth",
        "things": "raft-things.pth",
        "chairs": "raft-chairs.pth",
    }

    def __init__(self, model_name_descriptor='sintel', method="DeepFlow"):
        
        model_name = self.MODEL_NAME_OPTIONS.get(model_name_descriptor)
        
        if model_name is None:
            raise ValueError(f"Invalid model name descriptor: {model_name_descriptor}")
        
        if method == "RAFT":
            
            self.model = torch.nn.DataParallel(
                RAFT(args=instantiate_raft_model(model_name)))
            model_path = os.path.join(os.path.dirname(
                __file__), 'models\\' + model_name)
            if not os.path.exists(model_path):
                raise ValueError(
                    f"[ERROR] Model file '{model_name}' not found.")

            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.module
            self.model.to(DEVICE)
            self.model.eval()
            
        elif method == "DeepFlow":
            self.deepflow = cv2.optflow.createOptFlow_DeepFlow()

        self.coord_map = None
        self.coord_map_warped = None
        
    def fill_blank_areas(self, g_pos):
        # create a mask where the image is black
        mask = cv2.cvtColor(g_pos, cv2.COLOR_BGR2GRAY)
        mask[mask > 0] = 255  # You're setting non-black pixels to white; make sure this is your intention
        
        mask = cv2.bitwise_not(mask)  

        # seamless fill in the black areas using the surrounding areas in g_pos
        g_pos_inpaint = cv2.inpaint(g_pos, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
        
        
        # Show the original, mask, and inpainted images
        cv2.imshow("Original", g_pos)
        cv2.imshow("Mask", mask)
        cv2.imshow("Inpainted", g_pos_inpaint)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return g_pos_inpaint
    
    def create_and_warp_coord_map(self, flow_up, original_size):
        if self.coord_map is None:
            h, w = original_size[::-1]
            self.coord_map = torch.zeros((3, h, w)).to(DEVICE)
            self.coord_map[0] = torch.linspace(0, 1, w)
            self.coord_map[1] = torch.linspace(0, 1, h)[:, np.newaxis]
            self.coord_map_warped = self.coord_map.clone()
            return
        
        self.coord_map_warped = self.warp(self.coord_map_warped.unsqueeze(0), flow_up).squeeze(0)
        
    def create_g_pos_from_flow(self, flow_np, original_size):
        original_size = original_size
        g_pos_files = []
        for i in range(len(flow_np)+1):
            
            if i == 0:
                flow = flow_np[i]
            else:
                flow = flow_np[i-1]
                
            flow_up = torch.from_numpy(cv2.resize(flow, original_size[::-1])).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            self.create_and_warp_coord_map(flow_up, original_size)
            g_pos_np = self.create_g_pos_np(original_size)
            g_pos_files.append(g_pos_np)
            #self.fill_blank_areas(g_pos_np)

        return g_pos_files

    def create_g_pos_np(self, original_size):
        g_pos_np = self.coord_map_warped.cpu().numpy().transpose(1, 2, 0)
        g_pos_np = cv2.resize(g_pos_np, original_size)
        g_pos_np = np.clip(g_pos_np, 0, 1)
        g_pos_np = (g_pos_np * 255).astype(np.uint8)
        g_pos_np = cv2.cvtColor(g_pos_np, cv2.COLOR_BGR2RGB)
        #g_pos_np = cv2.GaussianBlur(g_pos_np, (10, 10), cv2.BORDER_DEFAULT)
        return g_pos_np

    def load_tensor_image(self, imfile):
        if isinstance(imfile, np.ndarray):
            img = imfile
        else:
            img = cv2.imread(imfile)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.bilateralFilter(img, 9, 100, 100)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor[None].to(DEVICE), img.shape[1::-1]

    def compute_optical_flow(self, image_batches, flow_output_dir=None, method="DeepFlow"):
        flow_results = []
        
        for image1, image2 in image_batches:
            image1_tensor, image1_size = self.load_tensor_image(image1)
            image2_tensor, image2_size = self.load_tensor_image(image2)
            
            if method == "RAFT":
                flow_result = self.compute_optical_flow_RAFT(
                    image1_tensor, image2_tensor, flow_output_dir, image1_size)
            elif method == "DeepFlow":
                flow_result = self.compute_optical_flow_deepflow(
                    image1_tensor, image2_tensor, flow_output_dir, image1_size)
                
            flow_results.append(flow_result)
        
        return flow_results

    def compute_optical_flow_RAFT(self, image1_tensor, image2_tensor, flow_output_dir=None, original_size=None):
        with torch.no_grad():
            padder = InputPadder(image1_tensor.shape)
            images = [padder.pad(image1_tensor)[0], padder.pad(image2_tensor)[0]]
            _, flow_up = self.model(images[0], images[1], iters=20, test_mode=True)

            flow_np = cv2.resize(flow_up[0].permute(1, 2, 0).cpu().numpy(), original_size)

            if flow_output_dir:
                np.save(os.path.join(flow_output_dir, 'input.npy'), flow_np)
                
            return flow_np

    def compute_optical_flow_deepflow(self, image1_tensor, image2_tensor, flow_output_dir=None, original_size=None):
        image1 = image1_tensor.squeeze(
            0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        image2 = image2_tensor.squeeze(
            0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        flow_np = self.deepflow.calc(image1_gray, image2_gray, None)

        if flow_output_dir:
            np.save(os.path.join(flow_output_dir, 'input.npy'), flow_np)

        return flow_np

    def warp(self, x, flo):
        """
        Warp an image or feature map with optical flow.

        :param x: Image or feature map to warp.
        :param flo: Optical flow.

        :return: Warped image or feature map.
        """
        try:
            B, C, H, W = x.size()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if x.is_cuda:
                grid = grid.cuda()

            flo = F.interpolate(flo, size=(
                H, W), mode='bilinear', align_corners=False)
            vgrid = grid + flo
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / \
                max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / \
                max(H - 1, 1) - 1.0
            vgrid = vgrid.permute(0, 2, 3, 1)
            output = F.grid_sample(x, vgrid)

            return output

        except Exception as e:
            print(f"[ERROR] Exception in warp: {e}")
            return None
