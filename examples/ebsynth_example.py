import cv2

from ezsynth.Ebsynth import Ebsynth, Config

ebsynth = Ebsynth()

config = Config(style_image = "input/000.jpg", guides = [("input/000.jpg", "styles/style000.jpg", 0.5)])

img, err = ebsynth(config)

cv2.imwrite("output/000.jpg", img)
