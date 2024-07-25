import gc
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.aux_classes import RunConfig
from ezsynth.aux_utils import save_to_folder, load_guide
from ezsynth.main_ez import ImageSynth

st = time.time()

output_folder = "J:/AI/Ezsynth/output_synth"

# Examples from the Ebsynth repository

## Segment retargetting

ezsynner = ImageSynth(
    style_path="J:/AI/Ezsynth/examples/texbynum/source_photo.png",
    src_path="J:/AI/Ezsynth/examples/texbynum/source_segment.png",
    tgt_path="J:/AI/Ezsynth/examples/texbynum/target_segment.png",
    cfg=RunConfig(img_wgt=1.0),
)

result = ezsynner.run()

save_to_folder(output_folder, "retarget_out.png", result[0])
save_to_folder(output_folder, "retarget_err.png", result[1])

## Stylit

ezsynner = ImageSynth(
    style_path="J:/AI/Ezsynth/examples/stylit/source_style.png",
    src_path="J:/AI/Ezsynth/examples/stylit/source_fullgi.png",
    tgt_path="J:/AI/Ezsynth/examples/stylit/target_fullgi.png",
    cfg=RunConfig(img_wgt=0.66),
)

result = ezsynner.run(
    guides=[
        load_guide(
            "J:/AI/Ezsynth/examples/stylit/source_dirdif.png",
            "J:/AI/Ezsynth/examples/stylit/target_dirdif.png",
            0.66,
        ),
        load_guide(
            "J:/AI/Ezsynth/examples/stylit/source_indirb.png",
            "J:/AI/Ezsynth/examples/stylit/target_indirb.png",
            0.66,
        ),
    ]
)

save_to_folder(output_folder, "stylit_out.png", result[0])
save_to_folder(output_folder, "stylit_err.png", result[1])

## Face style

ezsynner = ImageSynth(
    style_path="J:/AI/Ezsynth/examples/facestyle/source_painting.png",
    src_path="J:/AI/Ezsynth/examples/facestyle/source_Gapp.png",
    tgt_path="J:/AI/Ezsynth/examples/facestyle/target_Gapp.png",
    cfg=RunConfig(img_wgt=2.0),
)

result = ezsynner.run(
    guides=[
        load_guide(
            "J:/AI/Ezsynth/examples/facestyle/source_Gseg.png",
            "J:/AI/Ezsynth/examples/facestyle/target_Gseg.png",
            1.5,
        ),
        load_guide(
            "J:/AI/Ezsynth/examples/facestyle/source_Gpos.png",
            "J:/AI/Ezsynth/examples/facestyle/target_Gpos.png",
            1.5,
        ),
    ]
)

save_to_folder(output_folder, "facestyle_out.png", result[0])
save_to_folder(output_folder, "facestyle_err.png", result[1])

gc.collect()
torch.cuda.empty_cache()

print(f"Time taken: {time.time() - st:.4f} s")
