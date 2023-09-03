import os
import cv2
from ebsynth import stylizevideo

# Paths
STYLE_PATHS = ["STYLE_IMAGE_PATH_1", "STYLE_IMAGE_PATH_2", "STYLE_IMAGE_PATH_3"]  # List of style images
IMAGE_FOLDER = "PATH_TO_FRAMES"
OUTPUT_FOLDER = "PATH_TO_OUTPUT"

# Initialize and stylize
video_stylizer = stylizevideo(STYLE_PATHS, IMAGE_FOLDER)  # Pass the list of style images
video_stylizer.stylize(output_dir=OUTPUT_FOLDER)

print("[INFO] Stylized frames saved to:", OUTPUT_FOLDER)
