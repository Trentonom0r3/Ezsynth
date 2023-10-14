import numpy as np
import cProfile
import pstats
import time
from utils.utils import *
from utils.guides.guides import *
from memory_profiler import profile

@profile
def main():
    setup = Setup(STYLE_PATHS, IMAGE_FOLDER)
    run = Runner(setup)
    results = run.run()
    for i in range(len(results)):
        cv2.imwrite(f"{OUTPUT_FOLDER}/output{i}.jpg", results[i])
    

if __name__ == '__main__':
    STYLE_PATHS = [
        "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/output000.jpg",
        "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/output099.jpg",
    ]

    IMAGE_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input"
    OUTPUT_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output"
    
    main()
