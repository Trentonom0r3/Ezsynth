import unittest
import numpy as np
from utils.edge_detection import EdgeDetector
import os
import cv2

class TestEdgeDetector(unittest.TestCase):
    
    def setUp(self):
        self.edge_detector = EdgeDetector(method='Classic')
        self.image = np.random.rand(256, 256, 3) * 255
        self.file_path = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/blended_0.png"
        cv2.imwrite(self.file_path, self.image)
    
    def tearDown(self):
        os.remove(self.file_path)
    
    def test_load_image_from_file_path(self):
        image = self.edge_detector.load_image(self.file_path)
        self.assertEqual(image.shape, (256, 256, 3))
    
    def test_load_image_from_numpy_array(self):
        image = self.edge_detector.load_image(self.image)
        self.assertEqual(image.shape, (256, 256, 3))
    
    def test_create_gaussian_kernel(self):
        kernel = self.edge_detector.create_gaussian_kernel(size=5, sigma=6.0)
        self.assertEqual(kernel.shape, (5, 5))
    
    def test_edge_detection(self):
        edges = self.edge_detector.detect_edges(self.image)
        self.assertEqual(edges.shape, (256, 256))


    def test_edge_detection_method(self):
        edge_detector = EdgeDetector(method='PST')
        self.assertIsInstance(edge_detector.pst_gpu, edge_detector.method)
        edge_detector = EdgeDetector(method='PAGE')
        self.assertIsInstance(edge_detector.page_gpu, edge_detector.method)
        edge_detector = EdgeDetector(method='Classic')
        self.assertIsInstance(edge_detector.kernel, np.ndarray)