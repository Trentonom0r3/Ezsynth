# Import the EBSynth class from its module (assuming it's in the same directory)
import ebsynth
import cv2
# Define the paths to the images
style_image_path = "STYLE_IMAGE_PATH"
source_guide_path = "SOURCE_GUIDE_PATH"
target_guide_path = "TARGET_GUIDE_PATH"

# Create an instance of the EBSynth class
ebsynth = ebsynth.ebsynth(style=style_image_path, guides=[(source_guide_path, target_guide_path)])

# Run the style transfer
result_img = ebsynth.run()

# If you want to save or display the result:
cv2.imwrite("PATH_TO_YOUR_OUTPUT", result_img)
cv2.imshow("Styled Image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
