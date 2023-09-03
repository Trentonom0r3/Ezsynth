# Import the EBSynth class from its module (assuming it's in the same directory)
from ezsynth import ebsynth
import cv2
# Define the paths to the images
style_image_path = "C:/Users/tjerf/Downloads/000.jpg"
source_guide_path = "C:/Users/tjerf/Downloads/000.jpg"
target_guide_path = "C:/Users/tjerf/Downloads/000.jpg"

# Create an instance of the EBSynth class
ebsynth = ebsynth(style=style_image_path, guides=[(source_guide_path, target_guide_path, 1.0)])

# Run the style transfer
result_img = ebsynth.run()

# If you want to save or display the result:
cv2.imwrite("C:/Users/tjerf/Downloads/0001.jpg", result_img)
cv2.imshow("Styled Image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
