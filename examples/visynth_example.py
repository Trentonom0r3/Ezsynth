import cv2
import torch

from ezsynth.visynth import Visynth, Config, image_sequence_from_directory

visynth = Visynth()

config = Config(
    frames = image_sequence_from_directory("input"),
    style_frames = image_sequence_from_directory("styles"),
    edge_method = "classic",
    device = torch.device("cpu"),
)

frames = visynth(config)

for f in frames:
    cv2.imwrite("output/output" + str(f.index).zfill(3) + ".jpg", f.image)
