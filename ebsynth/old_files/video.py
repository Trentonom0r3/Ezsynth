from collections import defaultdict
import shutil
from threading import Lock
from ebsynth import ebsynth 
from concurrent.futures import ThreadPoolExecutor
import os
import re
import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.edge_detection import EdgeDetector
from utils.optical_flow import OpticalFlowProcessor
import concurrent.futures

class stylizevideo(ebsynth):
    """
    stylizevideo class is a specialized subclass of ebsynth for video stylization.
    It provides methods to process sequences of images.
    """
    
    def __init__(self, style, imgsequence, flow_method = 'DeepFlow', edge_method = 'PAGE'):
        """
        Initialize the stylizevideo instance.
        
        :param style: Path to the style image or a numpy array.
        :param imgsequence: List of image file paths or numpy arrays.
        :param flow_method: Optical flow method to use. Choice of 'RAFT' or 'DeepFlow' Defaults to 'DeepFlow'.
            :RAFT: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (RAFT) - Great performance, but can be slow.
            :DeepFlow: DeepFlow - Good performance, a bit faster than RAFT.
        :param edge_method: Edge detection method to use. Choice of 'Classic', 'PST', or 'PAGE'. Defaults to 'Classic'.
            :PST: Phase Stretch Transform (PST) edge detector. - Good overall structure, but not very detailed.
            :Classic: Classic edge detector. - A good balance between structure and detail.
            :PAGE: Phase and Gradient Estimation (PAGE) edge detector. - Great detail, great structure, but slow.
        """
        self.flow_method = flow_method
        self.edge_method = edge_method
            
        # Check if it's a folder path
        if os.path.isdir(imgsequence):
            # If it's a folder, list all image files inside it
            filenames = os.listdir(imgsequence)
            filenames.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', x)])
            self.imgsequence = [os.path.join(imgsequence, fname) 
                                for fname in filenames
                                if os.path.isfile(os.path.join(imgsequence, fname)) and 
                                fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            # If not, treat it as a list of images
            self.imgsequence = imgsequence
            
        self.edge_guides = self.compute_edge_guides(self.imgsequence)
        self.optical_flow = self.compute_optical_flow(self.imgsequence)
        
        if isinstance(style, list):
            self.style_indexes = self.extract_indexes(style)
            self.styles = self.load_style(style)  # Call the modified load_style method with the list of style paths
            self.sequences = self.split_sequence()  # Split the image sequence into multiple sequences
        else:
            self.style = self.load_style(style)  # Call the modified load_style method with the single style path
            super().__init__(style)
        self.lock = Lock()
        
    def process_sequences(self, output_dir=None, DEVICE='cuda'):
        """
        Process multiple sequences in parallel using multi-threading, including both forward and backward stylization.
        """

        def stylize_single_sequence(sequence, style_image, blend=False):
            img_subsequence, edge_subsequence, flow_subsequence = sequence[:3]  # Extract the first 3 elements

            if blend:
                img_subsequence = img_subsequence[::-1]
                edge_subsequence = edge_subsequence[::-1]
                flow_subsequence = flow_subsequence[::-1]

            stylized_sequence = self.stylize_sequence(style=style_image, imgsequence=img_subsequence,
                                                    edge_guides=edge_subsequence, optical_flow=flow_subsequence,
                                                    output_dir=output_dir, DEVICE=DEVICE, blend=blend)
            return stylized_sequence

        forward_results = []
        backward_results = []

        results_dict = defaultdict(lambda: [None, None])

        with ThreadPoolExecutor() as executor:
            futures = []
            for i, seq in enumerate(self.sequences):
                # Forward sequence
                style_image_forward = self.styles[i]
                futures.append((i, executor.submit(stylize_single_sequence, seq, style_image_forward, blend=False)))

                # Backward sequence
                if i < len(self.styles) - 1:
                    style_image_backward = self.styles[i + 1]
                    futures.append((i, executor.submit(stylize_single_sequence, seq, style_image_backward, blend=True)))

            # Collect results
            for i, future in enumerate(futures):
                index, result_future = future
                result = result_future.result()
                if result is not None:
                    results_dict[index][i % 2] = result

        final_results = []
        for index in sorted(results_dict.keys()):
            forward_sequence, backward_sequence = results_dict[index]
            final_results.extend(forward_sequence)
            if backward_sequence is not None:
                final_results.extend(backward_sequence[::-1])  # Add the reversed backward sequence

        print("[INFO] Processing of all sequences completed.")
        return final_results

        
    def split_sequence(self):
        """
        Split the image sequence, edge guides, and optical flow into multiple sequences based on the style indexes.

        :return: tuple containing [(imgsequence, edge_sequence, flow_sequence, style_image), ...]
        """
        sequences = []

        # Check if the last index in style_indexes is equal to the length of imgsequence - 1
        if self.style_indexes[-1] == len(self.imgsequence) - 1:
            # Remove the last index in style_indexes
            self.style_indexes.pop()

        for i in range(len(self.style_indexes)):
            start_idx = self.style_indexes[i]
            end_idx = self.style_indexes[i + 1] + 1 if i < len(self.style_indexes) - 1 else len(self.imgsequence)

            img_subsequence = self.imgsequence[start_idx:end_idx]
            edge_subsequence = self.edge_guides[start_idx:end_idx]
            flow_subsequence = self.optical_flow[start_idx:end_idx - 1]

            # Get the corresponding style image if styles is a list, or use the single style for all sequences
            style_corr = self.styles[i] if isinstance(self.styles, list) else self.style

            sequences.append((img_subsequence, edge_subsequence, flow_subsequence, style_corr))

        return sequences

    def extract_indexes(self, styles):
        """
        Extract the indexes from the style image file names.
        
        :param styles: List of style image file paths.
        
        :return: List of style indexes.
        """
        pattern = re.compile(r'(\d+)(?=\.(jpg|jpeg|png)$)')
        indexes = [int(pattern.findall(style)[-1][0]) for style in styles]
        return indexes
    
    def load_style(self, style):
        """
        Load the style image(s).

        :param style: Style path(s) as either a string or a list of strings.
        :return: Style image(s) as a numpy array or a list of numpy arrays.
        """
        if isinstance(style, list):
            styles = []
            for style_path in style:
                style_img = cv2.imread(style_path)
                if style_img is not None:
                    styles.append(style_img)
                else:
                    print(f"[ERROR] Failed to load style image: {style_path}")
                    return None
            return styles
        elif isinstance(style, str):
            style_img = cv2.imread(style)
            if style_img is not None:
                print(f"[INFO] Successfully loaded style image: {style}")
                return style_img
            else:
                print(f"[ERROR] Failed to load style image: {style}")
                return None
        else:
            raise ValueError("Style should be either a file path or a list of file paths.")
        
    def load_imgsequence(self):
        """
        Load image sequence into a list.
        
        :return: List of images as numpy arrays.
        """

        imgsequence = []
        print("[INFO] Loading image sequence...")
        for img in self.imgsequence:
            if isinstance(img, (str, np.ndarray)):
                imgsequence.append(cv2.imread(img) if isinstance(img, str) else img)
            else:
                raise ValueError("imgsequence should contain either file paths or numpy arrays.")
        
        return imgsequence 
    
    def compute_edge_guides(self, imgsequence):
        """
        Compute edge guides for the image sequence.
        
        :param imgsequence: List of images as numpy arrays.

        :return: List of edge guides as numpy arrays.
        """
        #compute edge guides for each image in imgsequence
        edge_guides = []
        edge_detector = EdgeDetector(method=self.edge_method)
        
        print(f"[INFO] Computing edge guides using {self.edge_method}...")
        
        for img in imgsequence:
            edge_guides.append(edge_detector.compute_edge(img))
            
        print(f"[INFO] Edge guides successfully computed using {self.edge_method}.")
        
        return edge_guides
    
    def compute_optical_flow(self, imgsequence):
        """
        Compute optical flow for the image sequence.

        :param imgsequence: List of images as numpy arrays.

        :return: List of optical flow results as numpy arrays.
        """
        optical_flow = []
        optical_flow_processor = OpticalFlowProcessor(model_name='raft-sintel.pth', method=self.flow_method)
        print(f"[INFO] Computing optical flow using {self.flow_method}...")
        # Create batches of image pairs
        image_batches = [(imgsequence[i], imgsequence[i+1]) for i in range(len(imgsequence)-1)]
        # Compute optical flow in parallel for the entire batch
        flow_results = optical_flow_processor.compute_optical_flow_parallel(image_batches, method=self.flow_method)
        # Collect and return results
        optical_flow.extend(flow_results)
        print(f"[INFO] Optical Flow successfully computed using {self.flow_method}.")
        return optical_flow
    
    def stylize(self, style=None, imgsequence=None, edge_guides=None,
                optical_flow=None, output_dir=None, DEVICE='cuda'):
            try:
                print("[INFO] Starting the stylization process...")
                # Check if multiple styles were provided
                if hasattr(self, 'styles') and isinstance(self.styles, list):
                    results = self.process_sequences(output_dir, DEVICE)  # Process sequences in parallel
                    return results
                else:
                    results = self.stylize_sequence(style= self.style, imgsequence=self.imgsequence,
                                                   edge_guides = self.edge_guides, optical_flow = self.optical_flow, 
                                                   output_dir = output_dir, DEVICE = DEVICE)  # Stylize using a single style
                    return results
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print(f"[ERROR] Exception encountered: {e}")
                return None
        
    def stylize_sequence(self, style=None, imgsequence=None, edge_guides=None, 
                         optical_flow=None, output_dir=None, DEVICE='cuda', blend = False):
        """
        Stylize each frame in the image sequence based on the style image and computed guides.

        :param output_dir: Directory to save intermediate results. If None, results won't be saved.
        :param DEVICE: PyTorch device ('CUDA' or 'CPU'). Default is 'CUDA'.

        :return: List of stylized images as numpy arrays.
        """
        with self.lock:
            try:
                optical_flow_processor = OpticalFlowProcessor(model_name='raft-sintel.pth', method=self.flow_method)
                # Initialize a list to store stylized images
                stylized_imgsequence = []
                for i in tqdm(range(len(imgsequence)), desc="Stylizing frames"):
                    # Collect guides
                    guides = [(edge_guides[0], edge_guides[i], 1), (imgsequence[0], imgsequence[i], 6)]
                    
                    # If it's not the first frame, get the warped stylized image from the previous iteration
                    if i >= 1:
                        stylized_image = stylized_imgsequence[-1]
                        # Convert to tensor and normalize
                        stylized_image = torch.from_numpy(stylized_image).permute(2, 0, 1).float() / 255.0
                        stylized_image = stylized_image.unsqueeze(0).to(DEVICE)
                        # Extract the optical flow and warp the stylized image
                        flow_np = optical_flow[i-1]
                        flow = torch.from_numpy(flow_np).permute(2, 0, 1).float()
                        flow = flow.unsqueeze(0).to(DEVICE)
                        if blend == False:
                            flow *= -1  # Invert the flow
                        elif blend == True:
                            flow = flow
                        else:
                            raise ValueError("blend should be either True or False.")
                        warped_image = optical_flow_processor.warp(stylized_image, flow)
                        warped_image_np = warped_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                        warped_image_np = np.clip(warped_image_np, 0, 1)  # Clip to [0,1]
                        warped_image_np = (warped_image_np * 255).astype(np.uint8)
                        guides.append((style, warped_image_np, 1.5))
                    
                    # Filter out any invalid (None) guides
                    valid_guides = [(src, tgt, wgt) for src, tgt, wgt in guides if src is not None and tgt is not None]
                    
                    # Initialize ebsynth with the valid guides
                    ebsynth_instance = ebsynth(style=style, guides=valid_guides, backend='cuda')

                    # Stylize the current frame
                    stylized_img = ebsynth_instance.run()
                    assert stylized_img is not None, f"[ERROR] Failed to stylize frame {i + 1}."
                    stylized_imgsequence.append(stylized_img)
            
                    if isinstance(style, str) and isinstance(output_dir, str):
                        # Save the output image
                        output_path = os.path.join(output_dir, 'outputs', f'stylized_frame_{i+1}.png')
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        cv2.imwrite(output_path, stylized_img)
                    else:
                        pass

                print("[INFO] Stylization process completed.")
                return stylized_imgsequence
            
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print(f"[ERROR] Exception encountered: {e}")
                return None




    



