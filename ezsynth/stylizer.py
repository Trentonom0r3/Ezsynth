import cv2
import torch
import numpy as np
import torch.nn.functional as F

from .utils import _create_selection_mask, _histogram_preserving_blending, _poisson_fusion
from .optical_flow import OpticalFlowProcessor
from .guide_classes import Guides
from . import ebsynth as eb


def warp(x, flo):
    """
    Warp an image or feature map with optical flow.

    :param x: Image to warp.
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

        flo = F.interpolate(flo, size=(H, W), mode='bilinear', align_corners=False)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid)

        return output

    except Exception as e:
        print(f"[ERROR] Exception in warp: {e}")
        return None

def perform_warping(last_stylized_img, step, i, flow_guides):
    stylized_image_bgr = cv2.cvtColor(last_stylized_img, cv2.COLOR_BGR2RGB)
    stylized_image = torch.from_numpy(stylized_image_bgr).permute(2, 0, 1).float() / 255.0
    stylized_image = stylized_image.unsqueeze(0)  # Removed .to('cuda')
    
    if step == 1:
        flow_up = cv2.resize(flow_guides[i-1], (stylized_image.shape[1::-1]))
        flow = torch.from_numpy(flow_up).permute(2, 0, 1).float().unsqueeze(0)  # Removed .to('cuda')
        flow *= -1
    elif step == -1:
        flow_up = cv2.resize(flow_guides[i], (stylized_image.shape[1::-1]))
        flow = torch.from_numpy(flow_up).permute(2, 0, 1).float().unsqueeze(0)
        
    warped_stylized = warp(stylized_image, flow)
    warped_stylized_np = warped_stylized.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    warped_stylized_np = np.clip(warped_stylized_np, 0, 1)
    warped_stylized_np = (warped_stylized_np * 255).astype(np.uint8)
    warped_stylized_np = cv2.cvtColor(warped_stylized_np, cv2.COLOR_RGB2BGR)

    return warped_stylized_np

def _stylize(start_idx, end_idx, style_start, style_end, g_pos_guides,
             g_pos_reverse, imgsequence, edge_guides, flow_guides):
    stylized_imgs = []
    nnf_list = []

    if start_idx == end_idx:
        raise ValueError("Start and End Indexes cannot be the same.")
    
    step = 1 if start_idx < end_idx else -1
    end_idx = end_idx + 1 if step == 1 else end_idx - 1  # Adjust the end index

    #ebsynth = eb.Ebsynth(style=style_start, guides=[])
    for i in range(start_idx, end_idx, step):
        ebsynth = eb.Ebsynth(style=style_start, guides=[])
            
        ebsynth.add_guide(imgsequence[start_idx], imgsequence[i], 6.0)

        ebsynth.add_guide(edge_guides[start_idx], edge_guides[i], 0.5)

        if step == 1:
            ebsynth.add_guide(g_pos_guides[start_idx], g_pos_guides[i], 2.0)
        elif step == -1:
            ebsynth.add_guide(g_pos_reverse[start_idx], g_pos_reverse[i], 2.0)

        if step == 1 and i != start_idx:
            last_stylized_img = stylized_imgs[-1]
            warped_stylized_np = perform_warping(last_stylized_img, step, i, flow_guides)
            ebsynth.add_guide(style_start, warped_stylized_np, 0.5)
        elif step == -1 and i != start_idx:
            last_stylized_img = stylized_imgs[-1]
            warped_stylized_np = perform_warping(last_stylized_img, step, i, flow_guides)
            ebsynth.add_guide(style_start, warped_stylized_np, 0.5)

        stylized_image, nnf = ebsynth.run(output_nnf=True)  # Assuming run_ebsynth is a valid function
        stylized_imgs.append(stylized_image)
        nnf_list.append(nnf)

    return stylized_imgs, nnf_list

def run_ebsynth(ebsynth):
    img, nnf = ebsynth.run(output_nnf=True)
    return img, nnf

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
        if self.style_start is None and self.style_end is None:
                raise ValueError("At least one style attribute should be provided.")
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
        self.flow = OpticalFlowProcessor(method='RAFT', model_name_descriptor='sintel')
    
    def _set_sequence(self):
        """
        Setup Sequence Information.
        Compares the style indexes with the image indexes to determine the sequence information.
        """
        sequences = []
        if self.num_styles == 1 and self.begFrame == self.style_indexes[0]:

            sequences.append(
                Sequence(begFrame= self.begFrame, endFrame= self.endFrame ,style_start = self.styles[0]))

            return sequences
        
        if self.style_indexes[0] == self.endFrame and self.num_styles == 1:

            sequences.append(
                Sequence(begFrame= self.begFrame , endFrame= self.endFrame, style_end = self.styles[0]))
        
            return sequences
        
        for i in range(self.num_styles-1):

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

    # if there is only one sequence, this loop will only run once
        for i, sequence in enumerate(self.sequences):
            start_idx = sequence.begFrame
            end_idx = sequence.endFrame
            style_start = sequence.style_start
            style_end = sequence.style_end

            if style_start is not None and style_end is not None:  # if both style attributes are provided

                # Call the _stylize method directly for forward stylization
                style_forward, err_forward = self._stylize(start_idx, end_idx, style_start, style_end)

                # Call the _stylize method directly for backward stylization
                style_backward, err_backward = self._stylize(end_idx, start_idx, style_end, style_start)

                # reverse the list, so that it's in ascending order
                style_backward = style_backward[::-1]
                err_backward = err_backward[::-1]
                SB = style_backward

                selection_masks = _create_selection_mask(
                    err_forward, err_backward)

                selection_masks = Guides.warp_masks(self, self.flow_guides, selection_masks)

                hpblended = _histogram_preserving_blending(style_forward, SB, selection_masks)

                final_blends = _poisson_fusion(hpblended, style_forward, SB, selection_masks)

                img_sequences.append(final_blends)

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

        # Adjust the end index to make sure it is included in the loop
        if step == 1:
            end_idx = end_idx + 1  # Add 1 to include end_idx when counting up
        else:
            end_idx = end_idx - 1  # Subtract 1 to include end_idx when counting down

        # Special case to make sure we don't go below zero when counting down
        if end_idx == 0 and step == -1:
            end_idx = -1  # This makes sure the loop includes 0 but doesn't go into negative numbers

        for i in range(start, end_idx, step):

            # Initialize ebsynth
            ebsynth = eb.Ebsynth(style=style, guides=[])

            ebsynth.add_guide(
                self.imgsequence[start], self.imgsequence[i], 6.0)

            ebsynth.add_guide(
                self.edge_guides[start], self.edge_guides[i], 0.5)

            if step == 1:
                ebsynth.add_guide(
                    self.g_pos_guides[start], self.g_pos_guides[i], 2.0)
            elif step == -1:
                ebsynth.add_guide(
                    self.g_pos_reverse[start], self.g_pos_reverse[i], 2.0)

            if i > (start) and i < (end_idx):

                stylized_image_bgr = cv2.cvtColor(
                    stylized_imgs[-1], cv2.COLOR_BGR2RGB)

                stylized_image = torch.from_numpy(
                    stylized_image_bgr).permute(2, 0, 1).float() / 255.0

                stylized_image = stylized_image.unsqueeze(0).to(self.DEVICE)

                flow = torch.from_numpy(
                    self.flow_guides[i-1]).permute(2, 0, 1).float().unsqueeze(0).to(self.DEVICE)

                flow *= -1

                warped_stylized = self.flow.warp(stylized_image, flow)

                warped_stylized_np = warped_stylized.squeeze(
                    0).permute(1, 2, 0).cpu().detach().numpy()


                warped_stylized_np = np.clip(warped_stylized_np, 0, 1)

                warped_stylized_np = (
                    warped_stylized_np * 255).astype(np.uint8)

                warped_stylized_np = cv2.cvtColor(
                    warped_stylized_np, cv2.COLOR_RGB2BGR)

                ebsynth.add_guide(style, warped_stylized_np, 0.5)


            elif i < (start) and i > (end_idx):
                #reverse works as expected, not sure why forward doesn't
                stylized_image_bgr = cv2.cvtColor(
                    stylized_imgs[-1], cv2.COLOR_BGR2RGB)
                stylized_image = torch.from_numpy(
                    stylized_image_bgr).permute(2, 0, 1).float() / 255.0

                stylized_image = stylized_image.unsqueeze(0).to(self.DEVICE)

                flow = torch.from_numpy(
                    self.flow_guides[i]).permute(2, 0, 1).float().unsqueeze(0).to(self.DEVICE)

                warped_stylized = self.flow.warp(stylized_image, flow)

                warped_stylized_np = warped_stylized.squeeze(
                    0).permute(1, 2, 0).cpu().detach().numpy()

                warped_stylized_np = np.clip(warped_stylized_np, 0, 1)

                warped_stylized_np = (
                    warped_stylized_np * 255).astype(np.uint8)

                warped_stylized_np = cv2.cvtColor(
                    warped_stylized_np, cv2.COLOR_RGB2BGR)

                ebsynth.add_guide(style, warped_stylized_np, 0.5)


            stylized_image, nnf = ebsynth.run(output_nnf=True)

            stylized_imgs.append(stylized_image)

            nnf_list.append(nnf)


        return stylized_imgs, nnf_list
    
    def run(self):
        """
        Run the stylization process.

        :return: A list of stylized images as numpy arrays.
        """

        stylized_imgs = self._process_sequences()

        return stylized_imgs