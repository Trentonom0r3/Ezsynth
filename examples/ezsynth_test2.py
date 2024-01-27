import os

import cv2

import ezsynth.utils.ezutils


def main():
    config, guides, sequences = ezsynth.utils.ezutils.setup(
        style_path = "styles",
        input_path = "input",
        edge_method = "Classic",
        flow_method = "RAFT",
        model_name = "sintel",
    )

    results = ezsynth.utils.ezutils.process(config, guides, sequences)

    output_folder = "output"
    os.makedirs(output_folder, exist_ok = True)

    for i, image in enumerate(results):
        cv2.imwrite(os.path.join(output_folder, "output" + str(i).zfill(3) + ".png"), image)


main()
