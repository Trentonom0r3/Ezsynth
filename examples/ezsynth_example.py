import os

import cv2

from ezsynth.utils.config import config_from_directory
from ezsynth.utils.ezutils import process
from ezsynth.utils.guides.guides import create_guides
from ezsynth.utils.sequences import SequenceManager

config = config_from_directory(
    style_path = "styles",
    input_path = "input",
    edge_method = "Classic",
    flow_method = "RAFT",
    model_name = "sintel",
)

guides = create_guides(config)

sequences = SequenceManager(config)._set_sequence()

images = process(config, guides, sequences)

output_folder = "output"
os.makedirs(output_folder, exist_ok = True)

for i, image in enumerate(images):
    cv2.imwrite(os.path.join(output_folder, "output" + str(i).zfill(3) + ".png"), image)
