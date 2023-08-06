import ebsynth
import cv2

# Define the paths to the images
style_image_path = "style.png"
guide_pairs = [
    ("guide1.png", "target2.png"),
    ("guide2.png", "target2.png"),
    ("guide3.png", "target3.png"),
    ("guide4.png", "target4.png")
]

# Create an instance of the EBSynth class with multiple guide pairs
ebsynth = ebsynth.ebsynth(style=style_image_path, guides=guide_pairs)

# Run the style transfer
result_img = ebsynth.run()

# If you want to save or display the result:
cv2.imwrite("output.png", result_img)
cv2.imshow("Styled Image with Multiple Guides", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
