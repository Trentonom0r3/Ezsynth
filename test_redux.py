import os
import sys

# import numpy as np
# import tqdm

import gc
import torch

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.aux_run import run_scratch
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow

from ezsynth.aux_computations import precompute_edge_guides
from ezsynth.aux_utils import (
    extract_indices,
    read_frames_from_paths,
    save_seq,
    setup_src_from_folder,
    validate_file_or_folder_to_lst,
)
from ezsynth.aux_classes import RunConfig
from ezsynth.utils.sequences import EasySequence, SequenceManager
from ezsynth.utils._ebsynth import ebsynth

st = time.time()

style_paths = [
    "J:/AI/Ezsynth/examples/styles/style000.png",
    # "J:/AI/Ezsynth/examples/styles/style002.png",
    # "J:/AI/Ezsynth/examples/styles/style003.png",
    # "J:/AI/Ezsynth/examples/styles/style006.png",
    # "J:/AI/Ezsynth/examples/styles/style014.png",
    # "J:/AI/Ezsynth/examples/styles/style019.png",
    # "J:/AI/Ezsynth/examples/styles/style099.jpg",
]

image_folder = "J:/AI/Ezsynth/examples/input"
output_folder = "J:/AI/Ezsynth/output"

# edge_method="Classic"
edge_method = "PAGE"
# edge_method="PST",
flow_method = "RAFT"
model = "sintel"

img_file_paths, img_idxes, img_frs_seq = setup_src_from_folder(image_folder)
style_paths = validate_file_or_folder_to_lst(style_paths, "style")

style_idxes = extract_indices(style_paths)
num_style_frs = len(style_paths)
style_frs = read_frames_from_paths(style_paths)

manager = SequenceManager(
    img_idxes[0],
    img_idxes[-1],
    style_paths,
    style_idxes,
    img_idxes,
)

sequences, atlas = manager.create_sequences()

edge_guides = precompute_edge_guides(img_frs_seq, edge_method)

stylized_frames = []

rafter = RAFT_flow(model_name="sintel")

cfg = RunConfig()

# cfg.only_mode = EasySequence.MODE_FWD
# cfg.only_mode = EasySequence.MODE_REV

eb = ebsynth(**cfg.get_ebsynth_cfg())
eb.runner.initialize_libebsynth()

num_seqs = len(sequences)
no_skip_rev = False

for i, seq in enumerate(sequences):
    if (
        cfg.only_mode == EasySequence.MODE_NON
        and i < num_seqs - 1
        and (atlas[i] == EasySequence.MODE_BLN or atlas[i + 1] != EasySequence.MODE_FWD)
    ):
        cfg.skip_blend_style_last = True
    else:
        cfg.skip_blend_style_last = False

    if (
        i > 0
        and cfg.only_mode == EasySequence.MODE_REV
        and atlas[i] == EasySequence.MODE_BLN
    ):
        seq.fr_start_idx += 1
        no_skip_rev = True

    tmp_stylized_frames, err_list = run_scratch(
        seq, img_frs_seq, style_frs, edge_guides, cfg, rafter, eb
    )

    if i > 0 and not no_skip_rev:
        if (atlas[i - 1] == EasySequence.MODE_REV) or (
            atlas[i - 1] == EasySequence.MODE_BLN
            and (
                atlas[i] == EasySequence.MODE_FWD
                or cfg.only_mode == EasySequence.MODE_FWD
            )
        ):
            tmp_stylized_frames.pop(0)
    no_skip_rev = False

    stylized_frames.extend(tmp_stylized_frames)


save_seq(stylized_frames, "J:/AI/Ezsynth/output_31")

gc.collect()
torch.cuda.empty_cache()

print(f"Time taken: {time.time() - st:.4f} s")
