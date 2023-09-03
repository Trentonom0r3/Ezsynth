# Import the EBSynth class from its module (assuming it's in the same directory)
import ebsynth as eb
import cv2
import time
# Define the paths to the images
style_image_path ="C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg"
guide_pairs = [
    ("C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg", "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg", 1.0),
    ("C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg","C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg", 3.0)
]

start = time.time()
# Create an instance of the EBSynth class with multiple guide pairs
ebsynth = eb.Ebsynth(style=style_image_path, guides=guide_pairs)
ebsynth.add_guide("C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg", "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg", 1.0)
result_img = ebsynth.run()
end = time.time()
print(end - start)
# If you want to save or display the result:
cv2.imshow("Styled Image with Multiple Guides", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
