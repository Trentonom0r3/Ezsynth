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

    for i in range(len(results)):
        save_results("output", "output" + str(i).zfill(3) + ".png", results[i])


def save_results(output_folder, base_file_name, a):
    os.makedirs(output_folder, exist_ok = True)
    output_file_path = os.path.join(output_folder, base_file_name)
    cv2.imwrite(output_file_path, a)
    return output_file_path


main()
