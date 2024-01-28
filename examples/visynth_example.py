import os

import cv2

from ezsynth.Visynth import Visynth, Config, image_sequence_from_directory

visynth = Visynth()

config = Config(
    frames = image_sequence_from_directory("input"),
    style_frames = image_sequence_from_directory("styles"),
)

images = visynth(config)

for i, image in enumerate(images):
    cv2.imwrite(os.path.join(output_folder, "output" + str(i).zfill(3) + ".png"), image)
