from ebsynth import Ebsynth, stylize
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import cProfile

STYLE = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/4.jpg")
SOURCE = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/000.jpg")
TARGET = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Input/000.jpg")

SOURCE2 = cv2.imread("C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/3.jpg")

# SOURCE is unaltered input
# Target is input with area removed by mask
# STYLE is stylized input, segmented to only show area that was removed from target


# STYLE is stylized input, segmented based on mask, pasted onto SOURCE with feathered edges
# SOURCE is unaltered input, with area removed by mask
# TARGET is unaltered input + 1, with area removed by mask


# for inpainting style = source_img
# guide = SOURCE, TARGET = what is to be inpainted (merged onto source with feathered)

GUIDES = [(SOURCE, TARGET, 1.0), (SOURCE2, TARGET, 1.0)]    
GUIDES2 = [(SOURCE, TARGET, 1.0)]

def test():
    eb = Ebsynth(style = STYLE, guides = [])
    eb.add_guide(SOURCE, TARGET, 1.0)
    t1 = time.time()
    for i in range(10):
        eb.run()
    t2 = time.time()
    print(f"Time: {t2 - t1}")
        
def main():
    eb = Ebsynth(style = STYLE, guides = [])
    eb.add_guide(SOURCE, TARGET, 1.0)
    ez = Ebsynth(style = STYLE, guides = [])
    ez.add_guide(SOURCE, TARGET, 1.0)
    with ProcessPoolExecutor(max_workers = 10) as executor:
        t1 = time.time()
        for i in range(5):
            img, nnf = executor.submit(eb.run, output_nnf=True).result()
            img2, nnf2 = executor.submit(ez.run, output_nnf=True).result()
        t2 = time.time()
        cv2.imshow("img", img)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Time: {t2 - t1}")
            
    # with ProcessPoolExecutor(max_workers = 10) as executor:
    #     for i in range(10):
    #         executor.submit(stylize, STYLE, SOURCE, TARGET, 1.0)
    
        
if __name__ == "__main__":
    main()
    print()
    test()
    