import os

import cv2

from ezsynth.Visynth import Visynth, Config, image_sequence_from_directory

config = Config(
    frames = image_sequence_from_directory("input"),
    style_frames = image_sequence_from_directory("styles"),
)

guides = create_guides(config)

sequences = SequenceManager(config)._set_sequence()

images = process(config, guides, sequences)

output_folder = "output"
os.makedirs(output_folder, exist_ok = True)

for i, image in enumerate(images):
    cv2.imwrite(os.path.join(output_folder, "output" + str(i).zfill(3) + ".png"), image)
