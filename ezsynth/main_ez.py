import time

from ezsynth.aux_classes import RunConfig
from ezsynth.aux_computations import precompute_edge_guides
from ezsynth.aux_run import run_scratch
from ezsynth.aux_utils import (
    extract_indices,
    read_frames_from_paths,
    setup_src_from_folder,
    validate_file_or_folder_to_lst,
)
from ezsynth.utils._ebsynth import ebsynth
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow
from ezsynth.utils.sequences import EasySequence, SequenceManager

EDGE_METHODS = ["PAGE", "PST", "Classic"]
DEFAULT_EDGE_METHOD = "Classic"

FLOW_MODELS = ["sintel", "kitti"]
DEFAULT_FLOW_MODEL = "sintel"


class Ezsynth:
    def __init__(
        self,
        style_paths: list[str],
        image_folder: str,
        cfg: RunConfig = RunConfig(),
        edge_method="Classic",
        raft_flow_model_name="sintel",
    ) -> None:
        st = time.time()

        self.img_file_paths, self.img_idxes, self.img_frs_seq = setup_src_from_folder(
            image_folder
        )
        style_paths = validate_file_or_folder_to_lst(style_paths, "style")
        self.style_idxes = extract_indices(style_paths)
        self.num_style_frs = len(style_paths)
        self.style_frs = read_frames_from_paths(style_paths)

        self.edge_method = (
            edge_method if edge_method in EDGE_METHODS else DEFAULT_EDGE_METHOD
        )
        self.flow_model = (
            raft_flow_model_name
            if raft_flow_model_name in FLOW_MODELS
            else DEFAULT_FLOW_MODEL
        )

        self.cfg = cfg

        manager = SequenceManager(
            self.img_idxes[0],
            self.img_idxes[-1],
            style_paths,
            self.style_idxes,
            self.img_idxes,
        )

        self.sequences, self.atlas = manager.create_sequences()
        self.num_seqs = len(self.sequences)

        self.edge_guides = precompute_edge_guides(self.img_frs_seq, self.edge_method)
        self.rafter = RAFT_flow(model_name=self.flow_model)

        self.eb = ebsynth(**cfg.get_ebsynth_cfg())
        self.eb.runner.initialize_libebsynth()

        print(f"Init Ezsynth took: {time.time() - st:.4f} s")

    def run_sequences(self, cfg_only_mode: str | None = None):
        st = time.time()

        if (
            cfg_only_mode is not None
            and cfg_only_mode in EasySequence.get_valid_modes()
        ):
            self.cfg.only_mode = cfg_only_mode

        no_skip_rev = False

        stylized_frames = []

        for i, seq in enumerate(self.sequences):
            if self._should_skip_blend_style_last(i):
                self.cfg.skip_blend_style_last = True
            else:
                self.cfg.skip_blend_style_last = False

            if self._should_rev_move_fr(i):
                seq.fr_start_idx += 1
                no_skip_rev = True

            tmp_stylized_frames, _ = run_scratch(
                seq,
                self.img_frs_seq,
                self.style_frs,
                self.edge_guides,
                self.cfg,
                self.rafter,
                self.eb,
            )

            if self._should_remove_first_fr(i, no_skip_rev):
                tmp_stylized_frames.pop(0)

            no_skip_rev = False

            stylized_frames.extend(tmp_stylized_frames)

        print(f"Run took: {time.time() - st:.4f} s")

        return stylized_frames

    def _should_skip_blend_style_last(self, i: int) -> bool:
        if (
            self.cfg.only_mode == EasySequence.MODE_NON
            and i < self.num_seqs - 1
            and (
                self.atlas[i] == EasySequence.MODE_BLN
                or self.atlas[i + 1] != EasySequence.MODE_FWD
            )
        ):
            return True
        return False

    def _should_rev_move_fr(self, i: int) -> bool:
        if (
            i > 0
            and self.cfg.only_mode == EasySequence.MODE_REV
            and self.atlas[i] == EasySequence.MODE_BLN
        ):
            return True
        return False

    def _should_remove_first_fr(self, i: int, no_skip_rev: bool) -> bool:
        if i > 0 and not no_skip_rev:
            if (self.atlas[i - 1] == EasySequence.MODE_REV) or (
                self.atlas[i - 1] == EasySequence.MODE_BLN
                and (
                    self.atlas[i] == EasySequence.MODE_FWD
                    or self.cfg.only_mode == EasySequence.MODE_FWD
                )
            ):
                return True
        return False
