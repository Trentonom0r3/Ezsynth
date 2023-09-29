import numpy as np
import cv2
import math

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
       
    def __str__(self):
        return f"Sequence: {self.begFrame} - {self.endFrame} | {self.style_start} - {self.style_end}"
         
class subsequence(Sequence):
    """
    Helper class for splitting sequences into sub-sequences.
    """
    CHUNK_SIZE = 10
    def __init__(self, begFrame, endFrame, style_start=None, style_end=None):
        super().__init__(begFrame, endFrame, style_start, style_end)
        self.subsequences = []
        self._split_sequence()
        
    def _split_sequence(self):
        """
        Split the sequence into sub-sequences of size CHUNK_SIZE.
        """
        num_chunks = math.ceil((self.endFrame - self.begFrame) / self.CHUNK_SIZE)
        for i in range(num_chunks):
            begFrame = self.begFrame + i * self.CHUNK_SIZE
            endFrame = min(begFrame + self.CHUNK_SIZE + 1, self.endFrame)
            self.subsequences.append(Sequence(begFrame, endFrame, self.style_start, self.style_end))
            
    def get_subsequences(self):
        return self.subsequences
    
    def __str__(self):
            return f"subsequence(begFrame={self.begFrame}, endFrame={self.endFrame}, style_start={self.style_start}, style_end={self.style_end}, subsequences=[{', '.join(str(s) for s in self.subsequences)}])"

class SequenceListManager:
    def __init__(self):
        self.sequence_lists = {}
        
    def add_sequence(self, list_name, sequence):
        if list_name not in self.sequence_lists:
            self.sequence_lists[list_name] = []
        self.sequence_lists[list_name].append(sequence)
        
    def get_subsequences(self, list_name):
        if list_name not in self.sequence_lists:
            return []
        
        subsequences = []
        for sequence in self.sequence_lists[list_name]:
            subseq = subsequence(sequence.begFrame, sequence.endFrame, sequence.style_start, sequence.style_end)
            subsequences.extend(subseq.get_subsequences())
        
        return subsequences
    
    def __str__(self):
        return str({k: [str(seq) for seq in v] for k, v in self.sequence_lists.items()})
    
class AdvancedSequence(Sequence):
    def __init__(self, begFrame, endFrame, style_start=None, style_end=None, 
                 guides=None, original_start=None, original_end=None, original_guides=None, unique_key=None):
        super().__init__(begFrame, endFrame, style_start, style_end)
        self.guides = guides if guides else {}
        self.original_start = original_start if original_start is not None else begFrame
        self.original_end = original_end if original_end is not None else endFrame
        self.original_guides = original_guides if original_guides else {}
        self.unique_key = unique_key 
        
    def __str__(self):
        return f"AdvancedSequence: {self.begFrame} - {self.endFrame} (Original: {self.original_start} - {self.original_end}) | {self.style_start} - {self.style_end} | Guides: {self.guides.keys()}"

class AdvancedSubsequence(subsequence):
    def __init__(self, begFrame, endFrame, style_start=None, style_end=None, guides=None, unique_key=None):
        super().__init__(begFrame, endFrame, style_start, style_end)
        self.guides = guides if guides else {}
        self.advanced_subsequences = []
        self.unique_key = unique_key  # Set unique_key for AdvancedSubsequence too
        self._split_advanced_sequence()

    def _split_advanced_sequence(self):
        num_chunks = math.ceil((self.endFrame - self.begFrame) / self.CHUNK_SIZE)
        for i, chunk_idx in enumerate(range(num_chunks)):
            begFrame = self.begFrame + chunk_idx * self.CHUNK_SIZE
            endFrame = min(begFrame + self.CHUNK_SIZE + 1, self.endFrame)

            # Slice guides for the subsequence
            sliced_guides = {k: v[begFrame:endFrame] for k, v in self.guides.items()}

            # Generate unique key for each AdvancedSequence
            sequence_unique_key = f"{self.unique_key}_{i}"  # Append index to the key

            # Create and append the AdvancedSequence
            self.advanced_subsequences.append(
                AdvancedSequence(
                    begFrame, endFrame, self.style_start, self.style_end, 
                    sliced_guides, original_start=self.begFrame, original_end=self.endFrame,
                    original_guides=self.guides, unique_key=sequence_unique_key
                )
            )

    def get_advanced_subsequences(self):
        return self.advanced_subsequences

    def __str__(self):
        return f"AdvancedSubsequence: {self.begFrame} - {self.endFrame} | {self.style_start} - {self.style_end} | Guides: {self.guides.keys()}"

class AdvancedSequenceListManager(SequenceListManager):
    def add_advanced_sequence(self, list_name, advanced_sequence):
        self.add_sequence(list_name, advanced_sequence)

    def get_advanced_subsequences(self, list_name):
        if list_name not in self.sequence_lists:
            return []

        advanced_subsequences = []
        for sequence in self.sequence_lists[list_name]:
            subseq = AdvancedSubsequence(sequence.begFrame, sequence.endFrame, sequence.style_start, sequence.style_end, sequence.guides)
            advanced_subsequences.extend(subseq.get_advanced_subsequences())

        return advanced_subsequences
