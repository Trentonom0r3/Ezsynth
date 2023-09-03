import os
import cv2
from ebsynth import stylizevideo  

# Paths
STYLE_PATH = "STYLE_IMAGE_PATH"
IMAGE_FOLDER = "PATH_TO_FRAMES"
OUTPUT_FOLDER = "PATH_TO_OUTPUT"

# Initialize and stylize
video_stylizer = stylizevideo(STYLE_PATH, IMAGE_FOLDER)
video_stylizer.stylize(output_dir=OUTPUT_FOLDER)

print("[INFO] Stylized frames saved to:", OUTPUT_FOLDER)

