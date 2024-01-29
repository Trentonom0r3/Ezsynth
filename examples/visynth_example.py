import cv2
import torch

from ezsynth.visynth import Visynth, Config, image_sequence_from_directory

visynth = Visynth()

frames, style_frames, frame_offset = image_sequence_from_directory(
    frames_directory = "input",
    style_frames_directory = "styles",
)

config = Config(
    frames = frames,
    style_frames = style_frames,
    edge_method = "classic",
    device = torch.device("cpu"),
)

frames = visynth(config)

for i, f in enumerate(frames):
    cv2.imwrite("output/output" + str(i + frame_offset).zfill(3) + ".jpg", f)
