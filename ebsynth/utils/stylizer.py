from concurrent.futures import ProcessPoolExecutor
import cv2
import torch
import time
import numpy as np
import ebsynth as eb
import torch
import torch.nn.functional as F
from utils.histogram_blend import HistogramBlender
from utils.gradient_blend import gradient_blending
from utils.optical_flow import OpticalFlowProcessor
from utils.poisson_blend import poisson_blend
from utils.reconstruction import ScreenedPoissonSolver

class Sequence:
    """
    Helper class to store sequence information.

    :param begFrame: Index of the first frame in the sequence.
    :param endFrame: Index of the last frame in the sequence.
    :param keyframeIdx: Index of the keyframe in the sequence.
    :param style_image: Style image for the sequence.

    :return: Sequence object.

    """

    def __init__(self, begFrame, endFrame, style_start=None, style_end=None):
        self.begFrame = begFrame
        self.endFrame = endFrame
        self.style_start = style_start if style_start else None
        self.style_end = style_end if style_end else None
        if self.style_start and self.style_end == None:
            raise ValueError(
                "At least one style attribute should be provided.")
        print(
            f"[INFO] Sequence: {self.begFrame}->{self.endFrame}, Style: {self.style_start}--{self.style_end}")
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Stylizer:
    
    def __init__(self, imgsequence, styles, style_indexes, flow_guides, edge_guides, g_pos_guides, g_pos_reverse, DEVICE):
        """
        Initialize the Stylizer class.

        :param imgsequence: List of images as numpy arrays.
        :param styles: List of style images as numpy arrays.
        :param style_indexes: List of indexes of the style images in the sequence.
        :param flow_guides: List of flow guides as numpy arrays.
        :param g_pos_guides: List of g_pos guides as numpy arrays.
        :param edge_guides: List of edge guides as numpy arrays.
        :param DEVICE: Device to run the stylization on.

        :return: Stylizer object.

        """

        self.imgsequence = imgsequence
        self.styles = styles
        self.style_indexes = style_indexes
        self.flow_guides = flow_guides
        self.edge_guides = edge_guides
        self.g_pos_guides = g_pos_guides
        self.g_pos_reverse = g_pos_reverse
        
        self.DEVICE = DEVICE

        self.num_styles = len(self.styles)
        self.num_frames = len(self.imgsequence)

        self.begFrame = 0
        self.endFrame = self.num_frames - 1

        self.sequences = self._set_sequence()

        # Initialize the Optical Flow object
        self.flow = OpticalFlowProcessor(self.DEVICE)
    
    def _set_sequence(self):
        """
        Setup Sequence Information.
        Compares the style indexes with the image indexes to determine the sequence information.
        """

        sequences = []
        for i in range(self.num_styles-1):
            if self.num_styles == 1:
                sequences.append(
                    Sequence(self.begFrame, self.endFrame, self.styles[i]))
                return sequences

            # If both style indexes are not None
            if self.style_indexes[i] is not None and self.style_indexes[i+1] is not None:
                if self.style_indexes[i] == self.begFrame and self.style_indexes[i+1] == self.endFrame:
                    sequences.append(
                        Sequence(self.begFrame, self.endFrame, self.styles[i], self.styles[i+1]))

                # If the first style index is the first frame in the sequence
                elif self.style_indexes[i] == self.begFrame and self.style_indexes[i+1] != self.endFrame:
                    sequences.append(Sequence(
                        self.begFrame, self.style_indexes[i+1], self.styles[i], self.styles[i+1]))

                # If the second style index is the last frame in the sequence
                elif self.style_indexes[i] != self.begFrame and self.style_indexes[i+1] == self.endFrame:
                    sequences.append(Sequence(
                        self.style_indexes[i], self.endFrame, self.styles[i], self.styles[i+1]))

                elif self.style_indexes[i] != self.begFrame and self.style_indexes[i+1] != self.endFrame and self.style_indexes[i] in self.imgindexes and self.style_indexes[i+1] in self.imgindexes:
                    sequences.append(Sequence(
                        self.style_indexes[i], self.style_indexes[i+1], self.styles[i], self.styles[i+1]))

        return sequences

    def _stylize(self, start_idx, end_idx, style_start, style_end):
        """
        Internal Method for Stylizing a Sequence.

        Parameters
        ----------
        start_idx : int
            Index of the first frame in the sequence.
        end_idx : int
            Index of the last frame in the sequence.
        style_start : str
            File path to the first style image in the sequence.
        style_end : str
            File path to the last style image in the sequence.
            
        Returns
        -------
        stylized_imgs : List of stylized images as numpy arrays.
        nnf_list : List of nearest neighbor fields as numpy arrays.
        
        """

        stylized_imgs = []
        nnf_list = []

        style = style_start  # Set the style image to the first style image in the sequence

        if start_idx == end_idx:
            raise ValueError("Start and End Indexes cannot be the same.")
        
        step = 1 if start_idx < end_idx else -1

        start = start_idx

        for i in range(start, end_idx, step):

            # Initialize ebsynth
            ebsynth = eb.Ebsynth(style=style, guides=[])

            ebsynth.add_guide(
                self.imgsequence[start], self.imgsequence[i], 4.0)

            ebsynth.add_guide(
                self.edge_guides[start], self.edge_guides[i], 1.0)

            if i > (start) and i <= (end_idx):

                stylized_image_bgr = cv2.cvtColor(
                    stylized_imgs[-1], cv2.COLOR_BGR2RGB)

                stylized_image = torch.from_numpy(
                    stylized_image_bgr).permute(2, 0, 1).float() / 255.0

                stylized_image = stylized_image.unsqueeze(0).to(self.DEVICE)

                flow = torch.from_numpy(
                    self.flow_guides[i-1]).permute(2, 0, 1).float().unsqueeze(0).to(self.DEVICE)

                #flow *= -1

                warped_stylized = self._warp(stylized_image, flow)

                warped_stylized_np = warped_stylized.squeeze(
                    0).permute(1, 2, 0).cpu().detach().numpy()

               
                warped_stylized_np = np.clip(warped_stylized_np, 0, 1)

                warped_stylized_np = (
                    warped_stylized_np * 255).astype(np.uint8)

                warped_stylized_np = cv2.cvtColor(
                    warped_stylized_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/warped" + str(i) + ".png", warped_stylized_np)
                ebsynth.add_guide(style, warped_stylized_np, 0.5)
                ebsynth.add_guide(
                    self.g_pos_guides[start], self.g_pos_guides[i-1], 1.5)

            elif i < (start) and i > (end_idx):
                
                stylized_image_bgr = cv2.cvtColor(
                    stylized_imgs[-1], cv2.COLOR_BGR2RGB)
                stylized_image = torch.from_numpy(
                    stylized_image_bgr).permute(2, 0, 1).float() / 255.0

                stylized_image = stylized_image.unsqueeze(0).to(self.DEVICE)

                flow = torch.from_numpy(
                    self.flow_guides[i-1]).permute(2, 0, 1).float().unsqueeze(0).to(self.DEVICE)

                warped_stylized = self._warp(stylized_image, flow)

                warped_stylized_np = warped_stylized.squeeze(
                    0).permute(1, 2, 0).cpu().detach().numpy()

                warped_stylized_np = np.clip(warped_stylized_np, 0, 1)

                warped_stylized_np = (
                    warped_stylized_np * 255).astype(np.uint8)

                warped_stylized_np = cv2.cvtColor(
                    warped_stylized_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/warpedrev" + str(i) + ".png", warped_stylized_np)
                ebsynth.add_guide(style, warped_stylized_np, 0.5)
                ebsynth.add_guide(
                    self.g_pos_reverse[start-1], self.g_pos_reverse[i-1], 1.5)

            stylized_image, nnf = ebsynth.run(output_nnf=True)
            
            stylized_imgs.append(stylized_image)

            nnf_list.append(nnf)

        return stylized_imgs, nnf_list

    def _warp(self, x, flo):
        """
        Warp an image or feature map with optical flow.

        :param x: Image or feature map to warp.
        :param flo: Optical flow.

        :return: Warped image or feature map.
        """
        DEVICE = 'cuda'
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
            mask = torch.ones(x.size()).to(DEVICE)
            mask = F.grid_sample(mask, vgrid)
            mask[mask < 0.999] = 0
            mask[mask > 0] = 1

            return output

        except Exception as e:
            print(f"[ERROR] Exception in warp: {e}")
            return None
    
    def _process_sequences(self):
        """
        Create Guides for Entire Sequence, using the data from Sequence objects.
        Sequence Objects contain information about the style images and the frames they will be applied to.
        begframe -> endframe - indexes of the frames in the imgsequence list
        style_start -> style_end - file paths

        TODO:
        :return: A list of Stylized Images. I.E. the final blended output for all sequences.
        """

        img_sequences = []  # list of stylized sequences, ascending order
        
        def _create_selection_mask(err_forward, err_backward):
            selection_masks = []

            for forward, backward in zip(err_forward, err_backward):
                # Check if the images have the same shape
                if forward.shape != backward.shape:
                    raise ValueError("Both images must have the same shape.")

                # Create a binary mask where the forward error metric is less than the backward error metric
                selection_mask = (forward < backward).astype(np.uint8)
      
                # Add to the list of masks
                selection_masks.append(selection_mask)

            return selection_masks

        def _histogram_preserving_blending(style_forward, style_backward, selection_masks):
            blended =[]
            for forward, backward, mask in zip(style_forward, style_backward, selection_masks):
                # Check if the images have the same shape
                if forward.shape[:2] != backward.shape[:2]:
                    raise ValueError("Both images must have the same shape.")

                blender = HistogramBlender()
                
                # Histogram matching step
                histogram_blend = blender.blend(forward, backward, mask)
                
                blended.append(histogram_blend)
        
            return blended
            
    # if there is only one sequence, this loop will only run once
        for i, sequence in enumerate(self.sequences):
            start_idx = sequence.begFrame
            end_idx = sequence.endFrame
            style_start = sequence.style_start
            style_end = sequence.style_end

            if style_start is not None and style_end is not None:  # if both style attributes are provided
                start = time.time()

                # Call the _stylize method directly for forward stylization
                style_forward, err_forward = self._stylize(start_idx, end_idx, style_start, style_end)

                # Call the _stylize method directly for backward stylization
                style_backward, err_backward = self._stylize(end_idx, start_idx, style_end, style_start)
                
                end = time.time()   
                print(f"Time taken: {end-start}")

                # reverse the list, so that it's in ascending order
                style_backward = style_backward[::-1]
                err_backward = err_backward[::-1]
                SB = style_backward

                selection_masks = _create_selection_mask(
                    err_forward, err_backward)
                
                
                hpblended = _histogram_preserving_blending(style_forward, SB, selection_masks)
                print("Blending")
                grad_x, grad_y = gradient_blending(style_forward, SB, selection_masks)
                print("Blended")
                blended = []
                for i in range(len(grad_x)):
                    slv = ScreenedPoissonSolver(hpblended[i], grad_x[i], grad_y[i])
                    solution = slv.run()
                    blended.append(solution)
                    cv2.imwrite("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/data" + str(i) + ".png", solution)
                    cv2.imwrite("C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/forward" + str(i) + ".png", style_forward[i])
                    
                img_sequences.append(blended)
                
            elif style_start is not None and style_end is None:

                style_forward, _ = self._stylize(
                    start_idx, end_idx, style_start, style_start)  # list of stylized images

                img_sequences.append(style_forward)

            elif style_start is None and style_end is not None:

                style_backward, _ = self._stylize(
                    end_idx, start_idx, style_end, style_end)

               
                style_backward = style_backward[::-1]

                img_sequences.append(style_backward)

            elif style_start is None and style_end is None:

                raise ValueError(
                    "At least one style attribute should be provided.")


        flattened_img_sequences = [img for sequence in img_sequences for img in sequence]

        return flattened_img_sequences  # single list containing all stylized images, ascending order
    
    def run(self):
        """
        Run the stylization process.

        :return: A list of stylized images as numpy arrays.
        """

        stylized_imgs = self._process_sequences()

        return stylized_imgs