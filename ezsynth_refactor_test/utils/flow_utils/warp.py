import cv2
import numpy as np
import torch
import torch.nn.functional as F

class Warp:
    def __init__(self, img):
        """
        Parameters
        ----------
        img : torch.Tensor
            Image to be warped. Shape: (B, C, H, W)
        
        Example
        -------
        >>> img = torch.rand(1, 3, 256, 256)
        >>> warp = SimpleWarp(img)
        >>> warped_img = warp.warp(img, flo)
        """
        print(f"img.shape: {img.dtype}")
        tensor_img = self._load_tensor_from_numpy(img)
        B, C, H, W = tensor_img.size()
        self.grid = self._create_grid(B, H, W)
        self.H = H
        self.W = W
        
    def _create_grid(self, B, H, W):
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        return grid
    
    def _warp(self, img, flo):
        """
        Warp an image or feature map with optical flow.
        
        Parameters
        ----------
        img : torch.Tensor
            Image to be warped. Shape: (B, C, H, W)
            flo : torch.Tensor
            Optical flow. Shape: (B, 2, H, W)
            
        Returns
        -------
        warped_img : torch.Tensor
        Warped image. Shape: (B, C, H, W)
        """
        try:
            flo = F.interpolate(flo, size=(self.H, self.W), mode='bilinear', align_corners=False)
            vgrid = self.grid + flo
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(self.W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(self.H - 1, 1) - 1.0
            vgrid = vgrid.permute(0, 2, 3, 1)
            output = F.grid_sample(img, vgrid)
            return output
        except Exception as e:
            print(f"[ERROR] Exception in warp: {e}")
            return None

    def _load_tensor_from_numpy(self, np_array):
        """
        Load a numpy array into a torch tensor.
        
        Parameters
        ----------
        np_array : numpy.ndarray
            Numpy array to be loaded into a torch tensor.
            
        Returns
        -------
        tensor : torch.Tensor
            Torch tensor loaded from the numpy array.
        """
        try:
            tensor = torch.from_numpy(np_array).permute(2, 0, 1).float().unsqueeze(0).to('cuda')
            return tensor
        except Exception as e:
            print(f"[ERROR] Exception in load_tensor_from_numpy: {e}")
            return None
        
    def _load_tensor_from_numpy_cpu(self, np_array):
        """
        Load a numpy array into a torch tensor.
        
        Parameters
        ----------
        np_array : numpy.ndarray
            Numpy array to be loaded into a torch tensor.
            
        Returns
        -------
        tensor : torch.Tensor
            Torch tensor loaded from the numpy array.
        """
        try:
            tensor = torch.from_numpy(np_array).permute(2, 0, 1).float().unsqueeze(0)
            return tensor
        except Exception as e:
            print(f"[ERROR] Exception in load_tensor_from_numpy: {e}")
            return None
        
    def _unload_tensor(self, tensor):
        """
        Unload a torch tensor into a numpy array.
        
        Parameters
        ----------
        tensor : torch.Tensor
            Torch tensor to be unloaded into a numpy array.
            
        Returns
        -------
        np_array : numpy.ndarray
            Numpy array unloaded from the torch tensor.
        """
        try:
            np_array = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            np_array = np.clip(np_array, 0, 1)
            np_array = (np_array * 255).astype(np.uint8)
            np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
            return np_array
        except Exception as e:
            print(f"[ERROR] Exception in unload_tensor: {e}")
            return None
        
    def run_warping_from_np(self, img, flow):
        """
        Run warping from numpy arrays.
        
        Parameters
        ----------
        img : numpy.ndarray
            Image to be warped.
        flow : numpy.ndarray
            Optical flow.
            
        Returns
        -------
        warped_img : numpy.ndarray
            Warped image.
        """
        try:
            img_tensor = self._load_tensor_from_numpy_cpu(img)
            flow_tensor = self._load_tensor_from_numpy_cpu(flow)
            print(f"type of img and flow: {type(img_tensor)} | {type(flow_tensor)}")
            warped_img_tensor = self._warp(img_tensor, flow_tensor)
            warped_img = self._unload_tensor(warped_img_tensor)
            return warped_img
        except Exception as e:
            print(f"[ERROR] Exception in run_warping_from_np: {e}")
            return None
        