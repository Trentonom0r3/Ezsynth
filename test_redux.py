import gc
import os
import sys
import time

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.sequences import EasySequence
from ezsynth.aux_classes import RunConfig
from ezsynth.aux_utils import save_seq
from ezsynth.main_ez import Ezsynth

st = time.time()

style_paths = [
    "J:/AI/Ezsynth/examples/styles/style000.jpg",
    # "J:/AI/Ezsynth/examples/styles/style002.png",
    # "J:/AI/Ezsynth/examples/styles/style003.png",
    # "J:/AI/Ezsynth/examples/styles/style006.png",
    "J:/AI/Ezsynth/examples/styles/style010.png",
    # "J:/AI/Ezsynth/examples/styles/style014.png",
    # "J:/AI/Ezsynth/examples/styles/style019.png",
    # "J:/AI/Ezsynth/examples/styles/style099.jpg",
]

image_folder = "J:/AI/Ezsynth/examples/input"
mask_folder = "J:/AI/Ezsynth/examples/mask/mask_feather"
output_folder = "J:/AI/Ezsynth/output"

# edge_method="Classic"
edge_method = "PAGE"
# edge_method="PST"
model = "sintel"

ezrunner = Ezsynth(
    style_paths=style_paths,
    image_folder=image_folder,
    cfg=RunConfig(pre_mask=False, feather=5),
    edge_method=edge_method,
    raft_flow_model_name=model,
    mask_folder=mask_folder,
    # do_mask=True,
    do_mask=False,
)


# only_mode = EasySequence.MODE_FWD
# only_mode = EasySequence.MODE_REV
only_mode = None

stylized_frames, err_frames = ezrunner.run_sequences(only_mode)

save_seq(stylized_frames, "J:/AI/Ezsynth/output_51")
# save_seq(err_frames, "J:/AI/Ezsynth/output_51err")

gc.collect()
torch.cuda.empty_cache()

print(f"Time taken: {time.time() - st:.4f} s")
