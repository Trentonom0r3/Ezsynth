import unittest
from utils.sequences import Sequence

class TestSequence(unittest.TestCase):
    def test_split_sequence(self):
        # Test case 1: chunk_size is greater than sequence length
        seq = Sequence(1, 10)
        with self.assertRaises(ValueError):
            seq.split_sequence(chunk_size=20, overlap=1)

        # Test case 2: chunk_size is less than 1
        seq = Sequence(1, 10)
        with self.assertRaises(ValueError):
            seq.split_sequence(chunk_size=0, overlap=1)

        # Test case 3: overlap is less than 0
        seq = Sequence(1, 10)
        with self.assertRaises(ValueError):
            seq.split_sequence(chunk_size=5, overlap=-1)

        # Test case 4: overlap is greater than or equal to chunk_size
        seq = Sequence(1, 10)
        with self.assertRaises(ValueError):
            seq.split_sequence(chunk_size=5, overlap=5)

        # Test case 5: sequence is split into chunks of size chunk_size with overlap
        seq = Sequence(1, 10)
        chunks = seq.split_sequence(chunk_size=3, overlap=1)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].begFrame, 1)
        self.assertEqual(chunks[0].endFrame, 3)
        self.assertEqual(chunks[1].begFrame, 3)
        self.assertEqual(chunks[1].endFrame, 5)
        self.assertEqual(chunks[2].begFrame, 5)
        self.assertEqual(chunks[2].endFrame, 7)
        self.assertEqual(chunks[3].begFrame, 7)
        self.assertEqual(chunks[3].endFrame, 10)

        # Test case 6: sequence is split into chunks of size chunk_size with overlap
        seq = Sequence(1, 10)
        chunks = seq.split_sequence(chunk_size=3, overlap=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].begFrame, 1)
        self.assertEqual(chunks[0].endFrame, 3)
        self.assertEqual(chunks[1].begFrame, 2)
        self.assertEqual(chunks[1].endFrame, 4)
        self.assertEqual(chunks[2].begFrame, 3)
        self.assertEqual(chunks[2].endFrame, 6)