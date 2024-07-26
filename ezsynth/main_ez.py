import time

import numpy as np

from ezsynth.aux_classes import RunConfig
from ezsynth.aux_computations import precompute_edge_guides
from ezsynth.aux_masker import (
    apply_masked_back_seq,
    apply_masks,
    apply_masks_idxes,
)
from ezsynth.aux_run import run_scratch
from ezsynth.aux_utils import (
    setup_masks_from_folder,
    setup_src_from_folder,
    setup_src_from_lst,
    validate_and_read_img,
    validate_option,
)
from ezsynth.constants import (
    DEFAULT_EDGE_METHOD,
    DEFAULT_FLOW_MODEL,
    EDGE_METHODS,
    FLOW_MODELS,
)
from ezsynth.utils._ebsynth import ebsynth
from ezsynth.utils.flow_utils.OpticalFlow import RAFT_flow
from ezsynth.utils.sequences import EasySequence, SequenceManager


class EzsynthBase:
    def __init__(
        self,
        style_frs: list[np.ndarray],
        style_idxes: list[int],
        img_frs_seq: list[np.ndarray],
        cfg: RunConfig = RunConfig(),
        edge_method="Classic",
        raft_flow_model_name="sintel",
        do_mask=False,
        msk_frs_seq: list[np.ndarray] | None = None,
    ) -> None:
        st = time.time()

        self.style_frs = style_frs
        self.style_idxes = style_idxes
        self.img_frs_seq = img_frs_seq
        self.msk_frs_seq = msk_frs_seq or []

        self.len_img = len(self.img_frs_seq)
        self.len_msk = len(self.msk_frs_seq)
        self.len_stl = len(self.style_idxes)

        self.msk_frs_seq = self.msk_frs_seq[: self.len_img]

        self.cfg = cfg
        self.edge_method = validate_option(
            edge_method, EDGE_METHODS, DEFAULT_EDGE_METHOD
        )
        self.flow_model = validate_option(
            raft_flow_model_name, FLOW_MODELS, DEFAULT_FLOW_MODEL
        )

        self.cfg.do_mask = do_mask and self.len_msk > 0
        print(f"Masking mode: {self.cfg.do_mask}")

        if self.cfg.do_mask and len(self.msk_frs_seq) != len(self.img_frs_seq):
            raise ValueError(
                f"Missing frames: Masks={self.len_msk}, Expected {self.len_img}"
            )

        self.style_masked_frs = None
        if self.cfg.do_mask and self.cfg.pre_mask:
            self.masked_frs_seq = apply_masks(self.img_frs_seq, self.msk_frs_seq)
            self.style_masked_frs = apply_masks_idxes(
                self.style_frs, self.msk_frs_seq, self.style_idxes
            )

        manager = SequenceManager(
            0,
            self.len_img - 1,
            self.len_stl - 1,
            self.style_idxes,
            list(range(0, self.len_img)),
        )

        self.sequences, self.atlas = manager.create_sequences()
        self.num_seqs = len(self.sequences)

        self.edge_guides = precompute_edge_guides(
            self.masked_frs_seq
            if (self.cfg.do_mask and self.cfg.pre_mask)
            else self.img_frs_seq,
            self.edge_method,
        )
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

        img_seq = (
            self.masked_frs_seq
            if (self.cfg.do_mask and self.cfg.pre_mask)
            else self.img_frs_seq
        )
        stl_seq = (
            self.style_masked_frs
            if (self.cfg.do_mask and self.cfg.pre_mask)
            else self.style_frs
        )

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
                img_seq,
                stl_seq,
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

        if self.cfg.do_mask and not self.cfg.return_masked_only:
            stylized_frames = apply_masked_back_seq(
                self.img_frs_seq, stylized_frames, self.msk_frs_seq, self.cfg.feather
            )

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
                    self.cfg.only_mode == EasySequence.MODE_FWD
                    or (
                        self.cfg.only_mode == EasySequence.MODE_REV
                        and self.atlas[i] == EasySequence.MODE_FWD
                    )
                )
            ):
                return True
        return False


class Ezsynth(EzsynthBase):
    def __init__(
        self,
        style_paths: list[str],
        image_folder: str,
        cfg: RunConfig = RunConfig(),
        edge_method="Classic",
        raft_flow_model_name="sintel",
        mask_folder: str | None = None,
        do_mask=False,
    ) -> None:
        _, _, img_frs_seq = setup_src_from_folder(image_folder)
        _, style_idxes, style_frs = setup_src_from_lst(style_paths, "style")
        msk_frs_seq = setup_masks_from_folder(mask_folder)[2] if do_mask else None

        super().__init__(
            style_frs=style_frs,
            style_idxes=style_idxes,
            img_frs_seq=img_frs_seq,
            cfg=cfg,
            edge_method=edge_method,
            raft_flow_model_name=raft_flow_model_name,
            do_mask=do_mask,
            msk_frs_seq=msk_frs_seq,
        )


class ImageSynth:
    def __init__(
        self,
        style_path: str,
        src_path: str,
        tgt_path: str,
        cfg: RunConfig = RunConfig(),
    ) -> None:
        self.style_img = validate_and_read_img(style_path)
        self.src_img = validate_and_read_img(src_path)
        self.tgt_img = validate_and_read_img(tgt_path)
        self.cfg = cfg

        self.eb = ebsynth(**cfg.get_ebsynth_cfg())
        self.eb.runner.initialize_libebsynth()

    def run(self, guides: list[tuple[np.ndarray, np.ndarray, float]] = []):
        guides.append((self.src_img, self.tgt_img, self.cfg.img_wgt))
        return self.eb.run(self.style_img, guides=guides)
