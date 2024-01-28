import cv2

from ezsynth.visynth import Visynth, Config, image_sequence_from_directory

visynth = Visynth()

config = Config(
    frames = image_sequence_from_directory("input"),
    style_frames = image_sequence_from_directory("styles"),
)

images = visynth(config)

for i, image in images:
    cv2.imwrite("output/output" + str(i).zfill(3) + ".jpg", image)
