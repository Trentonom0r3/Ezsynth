import numpy as np
import cv2
import time

from utils.utils import *
from utils.guides.guides import *


if __name__ == '__main__':
    STYLE_PATHS = [
        "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/output000.jpg",
        "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/output099.jpg",
    ]

    IMAGE_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input"
    OUTPUT_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output"

    setup = Setup(STYLE_PATHS, IMAGE_FOLDER)
    run = Runner(setup)
    results = run.run()

    

    