class EasySequence:
    MODE_FWD = "forward"
    MODE_REV = "reverse"
    MODE_BLN = "blend"

    def __init__(
        self, fr_start_idx: int, fr_end_idx: int, mode: str, style_idxs: list[int]
    ) -> None:
        self.fr_start_idx = fr_start_idx
        self.fr_end_idx = fr_end_idx
        self.mode = mode
        self.style_idxs = style_idxs

    def __repr__(self) -> str:
        return f"[{self.fr_start_idx}, {self.fr_end_idx}] {self.mode} {self.style_idxs}"


class SequenceManager:
    def __init__(self, begin_fr_idx, end_fr_idx, style_paths, style_idxs, img_idxs):
        self.begin_fr_idx = begin_fr_idx
        self.end_fr_idx = end_fr_idx
        self.style_paths = style_paths
        self.style_idxs = style_idxs
        self.img_idxs = img_idxs
        self.num_style_frs = len(style_paths)

    def create_sequences(self) -> list[EasySequence]:
        sequences = []

        # Handle sequence before the first style frame
        if self.begin_fr_idx < self.style_idxs[0]:
            sequences.append(
                EasySequence(
                    fr_start_idx=self.begin_fr_idx,
                    fr_end_idx=self.style_idxs[0],
                    mode=EasySequence.MODE_REV,
                    style_idxs=[0],
                )
            )

        # Handle sequences between style frames
        for i in range(len(self.style_idxs) - 1):
            sequences.append(
                EasySequence(
                    fr_start_idx=self.style_idxs[i],
                    fr_end_idx=self.style_idxs[i + 1],
                    mode=EasySequence.MODE_BLN,
                    style_idxs=[i, i + 1],
                )
            )

        # Handle sequence after the last style frame
        if self.end_fr_idx > self.style_idxs[-1]:
            sequences.append(
                EasySequence(
                    fr_start_idx=self.style_idxs[-1],
                    fr_end_idx=self.end_fr_idx,
                    mode=EasySequence.MODE_FWD,
                    style_idxs=[self.num_style_frs - 1],
                )
            )

        for seq in sequences:
            print(f"{seq}")
        return sequences