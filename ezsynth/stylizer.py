import time
import cv2
from .utils import _create_selection_mask, _stylize, final_blend
from .optical_flow import OpticalFlowProcessor
from .guide_classes import Guides
from .sequences import Sequence
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

        self.imgsequence = [cv2.imread(img) for img in imgsequence]
        self.styles = styles

        self.style_indexes = style_indexes
        self.flow_guides = flow_guides
        self.edge_guides = edge_guides
        self.g_pos_guides = g_pos_guides
        self.g_pos_reverse = g_pos_reverse
        
        self.guides = {
            'g_pos_guides': self.g_pos_guides,
            'g_pos_reverse': self.g_pos_reverse,
            'imgsequence': self.imgsequence,
            'edge_guides': self.edge_guides,
            'flow_guides': self.flow_guides
        }
        
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
           
    def _process_sequences(self, output_dir = None):
        """
        Create Guides for Entire Sequence, using the data from Sequence objects.
        Sequence Objects contain information about the style images and the frames they will be applied to.
        begframe -> endframe - indexes of the frames in the imgsequence list
        style_start -> style_end - file paths

        TODO:
        :return: A list of Stylized Images. I.E. the final blended output for all sequences.
        """

        img_sequences = []  # list of stylized sequences, ascending order
        
        #advanced_manager = AdvancedSequenceListManager()
    # if there is only one sequence, this loop will only run once
        for i, sequence in enumerate(self.sequences):
            print(f"Processing Sequence {i+1} of {len(self.sequences)}")
            
            start_idx = sequence.begFrame
            end_idx = sequence.endFrame
            style_start = sequence.style_start
            style_end = sequence.style_end
                
            gpos = self.g_pos_guides
            gpos_rev = self.g_pos_reverse
            imgseq = self.imgsequence
            edge = self.edge_guides
            flow = self.flow_guides
            
            if style_start is not None and style_end is not None:  # if both style attributes are provided

                style_forward, err_forward = _stylize(start_idx, end_idx, style_start, style_end, gpos,
                                                                        gpos_rev, imgseq, edge, flow, output_path=output_dir)
                                    
                style_backward, style_forward = _stylize( end_idx, start_idx, style_end, style_start, gpos,
                                                                        gpos_rev, imgseq, edge, flow, output_path=output_dir)
                    

                style_backward = style_backward[::-1]
                err_backward = err_backward[::-1]

                print("Stylization Complete, Blending...")
                selection_masks = _create_selection_mask(
                    err_forward, err_backward)
                
                selection_masks = Guides.warp_masks(self, self.flow_guides, selection_masks) 
                
                final_blends = final_blend(style_forward, style_backward, selection_masks)
                print("Finalizing Sequence...")
                img_sequences.append(final_blends)
                
            elif style_start is not None and style_end is None:

                style_forward, _ = _stylize(
                    start_idx, end_idx, style_start, style_start, gpos, gpos_rev, imgseq, edge, flow)
                print("Stylization Complete, finalizing sequence...")
                img_sequences.append(style_forward)

            elif style_end is not None and style_start is None:

                style_backward, _ = _stylize(
                    end_idx, start_idx, style_end, style_end, gpos, gpos_rev, imgseq, edge, flow)
        
                style_backward = style_backward[::-1]
                print("Stylization Complete, finalizing sequence...")
                img_sequences.append(style_backward)

            elif style_start is None and style_end is None:

                raise ValueError(
                    "At least one style attribute should be provided.")

        flattened_img_sequences = [img for sequence in img_sequences for img in sequence]

        return flattened_img_sequences  # single list containing all stylized images, ascending order
    
    def run(self, output_dir):
        """
        Run the stylization process.

        :return: A list of stylized images as numpy arrays.
        """

        stylized_imgs = self._process_sequences(output_dir)

        return stylized_imgs