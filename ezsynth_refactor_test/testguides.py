import unittest
import numpy as np
from utils.guides.guides import GuideFactory

class TestGuideFactory(unittest.TestCase):
    def setUp(self):
        self.imgsequence = [np.zeros((100, 100, 3)) for _ in range(5)]
        self.factory = GuideFactory(self.imgsequence)
        
    def test_create_all_guides(self):
        guides = self.factory.create_all_guides()
        self.assertEqual(len(guides), 3)
        self.assertIn("edge", guides)
        self.assertIn("flow", guides)
        self.assertIn("positional", guides)
        
    def test_add_custom_guide(self):
        custom_guides = [np.zeros((100, 100, 3)) for _ in range(5)]
        self.factory.add_custom_guide("custom", custom_guides)
        self.assertIn("custom", self.factory.guides)
        self.assertEqual(self.factory.guides["custom"], custom_guides)
        
    def test_add_custom_guide_wrong_length(self):
        custom_guides = [np.zeros((100, 100, 3)) for _ in range(4)]
        with self.assertRaises(ValueError):
            self.factory.add_custom_guide("custom", custom_guides)