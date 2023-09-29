from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from . import ebsynth as eb
from .reconstruction import poisson_fusion
from .histogram_blend import HistogramBlender
from .guide_classes import Guides

def _create_selection_mask(err_forward, err_backward):
    selection_masks = []

    for forward, backward in zip(err_forward, err_backward):
        # Check if the images have the same shape
        if forward.shape != backward.shape:
            raise ValueError("Both images must have the same shape.")

        # Create a binary mask where the forward error metric is less than the backward error metric
        selection_mask = np.where( forward < backward, 0, 1)
        selection_mask = selection_mask.astype(np.uint8)

        # Add to the list of masks
        selection_masks.append(selection_mask)

    return selection_masks

# Sample chunk function for histogram preserving blending
def _histogram_preserving_blending_chunk(chunk):
    blended = []
    for forward, backward, mask in chunk:
        if forward.shape[:2] != backward.shape[:2]:
            raise ValueError("Both images must have the same shape.")
        blender = HistogramBlender()
        histogram_blend = blender.blend(forward, backward, mask)
        blended.append(histogram_blend)
    return blended

# Full function for histogram preserving blending
def _histogram_preserving_blending(style_forward, style_backward, selection_masks):
    num_threads = os.cpu_count()
    if num_threads > len(style_forward) or len(style_forward) == 0:
        num_threads = max(1, len(style_forward))
    chunk_size = len(style_forward) // num_threads
    if chunk_size == 0:
        chunk_size = 1

    chunks = [
        list(zip(style_forward[i:i+chunk_size], style_backward[i:i+chunk_size], selection_masks[i:i+chunk_size]))
        for i in range(0, len(style_forward), chunk_size)
    ]

    blended_result = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_histogram_preserving_blending_chunk, chunk) for chunk in chunks]
        for future in futures:
            blended_chunk = future.result()
            blended_result.extend(blended_chunk)

    return blended_result

# Sample chunk function for poisson fusion
def _poisson_fusion_chunk(chunk):
    blended = []
    for hpblended, forward, backward, mask in chunk:
        if forward.shape[:2] != backward.shape[:2]:
            raise ValueError("Both images must have the same shape.")
        poisson_blend = poisson_fusion(hpblended, forward, backward, mask)
        blended.append(poisson_blend)
    return blended

def _poisson_fusion(hpblended, style_forward, style_backward, selection_masks):
    num_threads = os.cpu_count()
    if num_threads > len(hpblended) or len(hpblended) == 0:
        num_threads = max(1, len(hpblended))
    chunk_size = len(hpblended) // num_threads
    if chunk_size == 0:
        chunk_size = 1

    chunks = [
        list(zip(hpblended[i:i+chunk_size], style_forward[i:i+chunk_size], style_backward[i:i+chunk_size], selection_masks[i:i+chunk_size]))
        for i in range(0, len(hpblended), chunk_size)
    ]

    blended_result = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_poisson_fusion_chunk, chunk) for chunk in chunks]
        for future in futures:
            blended_chunk = future.result()
            blended_result.extend(blended_chunk)
    
    return blended_result

def final_blend(style_fwd, style_bwd, selection_masks):
    hpblended = _histogram_preserving_blending(style_fwd, style_bwd, selection_masks)
                    
    final_blends = _poisson_fusion(hpblended, style_fwd, style_bwd, selection_masks)
    
    return final_blends


def warp(x, flo):
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

def perform_warping(last_stylized_img, step, i, flow_guides):
    stylized_image_bgr = cv2.cvtColor(last_stylized_img, cv2.COLOR_BGR2RGB)
    stylized_image = torch.from_numpy(stylized_image_bgr).permute(2, 0, 1).float() / 255.0
    stylized_image = stylized_image.unsqueeze(0).to('cuda')  # Removed .to('cuda')

    if step == 1:
        flow = torch.from_numpy(flow_guides[i-1]).permute(2, 0, 1).float().unsqueeze(0).to('cuda') # Removed .to('cuda')
        flow *= -1
    else:
        flow = torch.from_numpy(flow_guides[i]).permute(2, 0, 1).float().unsqueeze(0).to('cuda')
        
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

    ebsynth = eb.Ebsynth(style=style_start, guides=[])
    for i in range(start_idx, end_idx, step):
        ebsynth.clear_guides()
            
        ebsynth.add_guide(imgsequence[start_idx], imgsequence[i], 6.0)

        ebsynth.add_guide(edge_guides[start_idx], edge_guides[i], 0.5)

        if step == 1:
            ebsynth.add_guide(g_pos_guides[start_idx], g_pos_guides[i], 2.0)
        elif step == -1:
            ebsynth.add_guide(g_pos_reverse[start_idx], g_pos_reverse[i], 2.0)

        if i != start_idx and i != end_idx:  # Skip the first and last iterations
            last_stylized_img = stylized_imgs[-1]
            warped_stylized_np = perform_warping(last_stylized_img, step, i, flow_guides)
            ebsynth.add_guide(style_start, warped_stylized_np, 0.5)

        stylized_image, nnf = ebsynth.run(output_nnf=True)  # Assuming run_ebsynth is a valid function
        stylized_imgs.append(stylized_image)
        nnf_list.append(nnf)

    return stylized_imgs, nnf_list

    
class runner:
    def __init__(self, style_img, guides, original_guides):
        self.style_img = style_img # single image
        self.imgseq = guides['imgsequence'] # list of images
        self.edges = guides['edge_guides'] # list of images
        self.g_pos = guides['g_pos_guides'] # list of images
        self.flow = guides['flow_guides'] # list of images
        self.img_init = original_guides['imgsequence'][0] # single image
        self.edge_init = original_guides['edge_guides'][0] # single image
        self.g_pos_init = original_guides['g_pos_guides'][0]
        self.eb = self._init_ebsynth(self.style_img)
    
    def _init_ebsynth(self, style_img):
        """returns an ebsynth object with the style image set"""
        ebsynth = eb.Ebsynth(style=style_img, guides=[])
        return ebsynth
    
    def set_guides(self, i):
        """sets the guides for the ebsynth object"""
        self.eb.clear_guides()  # reset guides
        self.eb.add_guide(self.img_init, self.imgseq[i], 6.0)
        self.eb.add_guide(self.edge_init, self.edges[i], 0.5)
        self.eb.add_guide(self.g_pos_init, self.g_pos[i], 2.0)
        return self.eb
    
    def add_warped(self, style_img, warped_stylized):
        self.eb.add_guide(style_img, warped_stylized, 0.5)
        return self.eb
    
    def run(self):
        stylized_image, nnf = self.eb.run(output_nnf=True)
        return stylized_image, nnf
    
class batch(runner):
    def __init__(self, style_img, guides, original_guides, reverse = False):
        super().__init__(style_img, guides, original_guides)
        self.start = 0 if reverse == False else len(self.imgseq)
        self.end = len(self.imgseq) if reverse == False else 0
        self.stylized_imgs = []
        self.nnf_list = []
    
    def _perform_warping(self, step, i):
        stylized_image_bgr = cv2.cvtColor(self.stylized_imgs[-1], cv2.COLOR_BGR2RGB)
        stylized_image = torch.from_numpy(stylized_image_bgr).permute(2, 0, 1).float() / 255.0
        stylized_image = stylized_image.unsqueeze(0).to('cuda')
        if step == 1:
            flow = torch.from_numpy(self.flow[i-1]).permute(2, 0, 1).float().unsqueeze(0).to('cuda')
            flow *= -1
        else:
            flow = torch.from_numpy(self.flow[i]).permute(2, 0, 1).float().unsqueeze(0).to('cuda')
        warped_stylized = warp(stylized_image, flow)
        warped_stylized_np = warped_stylized.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        warped_stylized_np = np.clip(warped_stylized_np, 0, 1)
        warped_stylized_np = (warped_stylized_np * 255).astype(np.uint8)
        warped_stylized_np = cv2.cvtColor(warped_stylized_np, cv2.COLOR_RGB2BGR)
        return warped_stylized_np
    
    def run(self):
        step = 1 if self.start < self.end else -1
        for i in range(self.start, self.end, step):
            self.set_guides(i)
            if i != self.start and i != self.end:
                warped_stylized_np = self._perform_warping(step, i)
                self.add_warped(self.style_img, warped_stylized_np)
            stylized_image, nnf = self.eb.run(output_nnf=True)
            self.stylized_imgs.append(stylized_image)
            self.nnf_list.append(nnf)

        return self.stylized_imgs, self.nnf_list
    
class overlap(batch):
    OVERLAP = 1.0
    def __init__(self, style_img, guides1, guides2, 
                 original_guides1, original_guides2, reverse = False):
        self.batch1 = batch(style_img, guides1, original_guides1, reverse)
        self.batch2 = batch(style_img, guides2, original_guides2, reverse)
        self.stylized_imgs1 = []
        self.stylized_imgs2 = []
        self.nnf_list1 = []
        self.nnf_list2 = []
        
    def run(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.batch1.run)
            future2 = executor.submit(self.batch2.run)
            self.stylized_imgs1, self.nnf_list1 = future1.result()  
            self.stylized_imgs2, self.nnf_list2 = future2.result()

        self._overlap()
        return self.stylized_imgs, self.nnf_list
    
    def _overlap(self):
        # take the last OVERLAP frames from batch1 and the first OVERLAP frames from batch2
        # blend them together
        # add the blended frames to batch1 and batch2
        # return flattened lists of stylized images and nnf lists
        print(f"OVERLAP" + str(self.OVERLAP))
        overlap1 = self.stylized_imgs1[-int(self.OVERLAP):]
        overlap2 = self.stylized_imgs2[:int(self.OVERLAP)]
        err1 = self.nnf_list1[-int(self.OVERLAP):]
        err2 = self.nnf_list2[:int(self.OVERLAP)]
        flow = self.batch1.flow[-int(self.OVERLAP):]
        new_err = []
        for i in range(len(err1)):
            new_err.append(np.where(err1[i] < err2[i], err1[i], err2[i]))
        selection_masks = _create_selection_mask(err1, err2)
        selection_masks = Guides().warp_masks(flow, selection_masks)
        final_blends = final_blend(overlap1, overlap2, selection_masks)

        #replace the overlap frames in batch 1, remove the overlap frames in batch 2
        self.stylized_imgs1 = self.stylized_imgs1[:-int(self.OVERLAP)] + final_blends
        self.stylized_imgs2 = self.stylized_imgs2[int(self.OVERLAP):]

        self.nnf_list1 = self.nnf_list1[:-int(self.OVERLAP)] + new_err
        self.nnf_list2 = self.nnf_list2[int(self.OVERLAP):]
        
        # flatten lists to return a combined list stylized images and nnf lists
        self.stylized_imgs = self.stylized_imgs1 + self.stylized_imgs2
        self.nnf_list = self.nnf_list1 + self.nnf_list2

        return self.stylized_imgs, self.nnf_list
    
class BatchProcessor:
    def __init__(self, advanced_subsequences, style_img, OVERLAP=1.0):
        self.advanced_subsequences = advanced_subsequences
        self.style_img = style_img
        self.OVERLAP = OVERLAP
        self.img_chunks = []
        self.nnf_chunks = []
    
    def run_batches(self):
        with ThreadPoolExecutor(max_workers=len(self.advanced_subsequences)) as executor:
            futures = [executor.submit(batch(self.style_img, subseq.guides, subseq.original_guides).run) 
                       for subseq in self.advanced_subsequences]

        for future in futures:
            imgs, nnf = future.result()
            self.img_chunks.append(imgs)
            self.nnf_chunks.append(nnf)
    
    def blend_overlaps(self):
        for i in range(len(self.img_chunks) - 1):
            overlap1 = self.img_chunks[i][-int(self.OVERLAP):]
            overlap2 = self.img_chunks[i + 1][:int(self.OVERLAP)]
            err1 = self.nnf_chunks[i][-int(self.OVERLAP):]
            err2 = self.nnf_chunks[i + 1][:int(self.OVERLAP)]
            flow = self.batch1.flow[-int(self.OVERLAP):]
            new_err = []
            for i in range(len(err1)):
                new_err.append(np.where(err1[i] < err2[i], err1[i], err2[i]))
            selection_masks = _create_selection_mask(err1, err2)
            selection_masks = Guides().warp_masks(flow, selection_masks)
            final_blends = final_blend(overlap1, overlap2, selection_masks)

            # Replace the overlap frames in batch 1 and remove the overlap frames in batch 2
            self.img_chunks[i] = self.img_chunks[i][:-int(self.OVERLAP)] + final_blends
            self.img_chunks[i + 1] = self.img_chunks[i + 1][int(self.OVERLAP):]

            self.nnf_chunks[i] = self.nnf_chunks[i][:-int(self.OVERLAP)] + new_err
            self.nnf_chunks[i + 1] = self.nnf_chunks[i + 1][int(self.OVERLAP):]
        
        # Flatten lists to return combined lists of stylized images and nnf lists
        self.final_imgs = sum(self.img_chunks, [])
        self.final_nnf = sum(self.nnf_chunks, [])
        
    
        

