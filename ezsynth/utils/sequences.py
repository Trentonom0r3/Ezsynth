class Sequence:
    def __init__(
        self,
        begin_fr_idx: int,
        end_fr_idx: int,
        style_start_fr: str | None = None,
        style_end_fr: str | None = None,
        is_all=True,
        is_blend=False
    ):
        self.begin_fr_idx = begin_fr_idx
        self.end_fr_idx = end_fr_idx

        self.style_start_fr = style_start_fr
        self.style_end_fr = style_end_fr

        self.init_idx = begin_fr_idx
        self.final_idx = end_fr_idx

        self.miss_start_style = self.style_start_fr is None
        self.miss_end_style = self.style_end_fr is None
        self.is_all = is_all
        self.is_blend = is_blend

        if self.miss_start_style is None and self.miss_end_style:
            raise ValueError("At least one style frame needed")

    def __str__(self):
        start_info = self.style_start_fr if not self.miss_start_style else "None"
        end_info = self.style_end_fr if not self.miss_end_style else "None"
        return f"Sequence: {self.begin_fr_idx} - {self.end_fr_idx} | Style Start: {start_info} - Style End: {end_info}"

    # def reversed(self):
    #     return Sequence(
    #         begin_fr_idx=self.end_fr_idx,
    #         end_fr_idx=self.begin_fr_idx,
    #         style_start_fr=self.style_end_fr,
    #         style_end_fr=self.style_start_fr,
    #     )


class Subsequence:
    def __init__(
        self,
        init_fr_idx: int,
        final_fr_idx: int,
        begin_fr_idx: int,
        end_fr_idx: int,
        style_start_fr: str | None = None,
        style_end_fr: str | None = None,
    ):
        self.init_fr_idx = init_fr_idx
        self.final_fr_idx = final_fr_idx
        self.begin_fr_idx = begin_fr_idx
        self.end_fr_idx = end_fr_idx
        self.style_start_fr = style_start_fr
        self.style_end_fr = style_end_fr
        self.miss_start_style = self.style_start_fr is None
        self.miss_end_style = self.style_end_fr is None

        if self.miss_start_style is None and self.miss_end_style:
            raise ValueError("At least one style frame needed")

    def __str__(self):
        return (
            f"Subsequence: Init: {self.init_fr_idx} - {self.final_fr_idx} | "
            f"Range: {self.begin_fr_idx} - {self.end_fr_idx} | Styles: {self.style_start_fr} - {self.style_end_fr}"
        )


class SequenceManager:
    def __init__(self, begin_fr_idx, end_fr_idx, style_paths, style_idxs, img_idxs):
        self.begin_fr_idx = begin_fr_idx
        self.end_fr_idx = end_fr_idx
        self.style_paths = style_paths
        self.style_idxs = style_idxs
        self.img_idxs = img_idxs
        self.num_style_frs = len(style_paths)

    def _set_sequence(self) -> list[Sequence]:
        """
        Setup Sequence Information.
        Compares the style indexes with the image indexes to determine the sequence information.
        """
        sequences: list[Sequence] = []
        print(f"{self.style_idxs=}")
        if self.num_style_frs == 1:
            # If first style is first frame
            if self.style_idxs[0] == self.begin_fr_idx:
                sequences.append(
                    Sequence(
                        begin_fr_idx=self.begin_fr_idx,
                        end_fr_idx=self.end_fr_idx,
                        style_start_fr=self.style_paths[0],
                        is_all=True,
                        is_blend=False
                    )
                )

                return sequences
            # If first style is last frame
            if self.style_idxs[0] == self.end_fr_idx:
                sequences.append(
                    Sequence(
                        begin_fr_idx=self.begin_fr_idx,
                        end_fr_idx=self.end_fr_idx,
                        style_end_fr=self.style_paths[0],
                        is_all=True,
                        is_blend=False
                    )
                )
                return sequences

        for i in range(self.num_style_frs - 1):
            # Both style must not be None
            if self.style_idxs[i] is None or self.style_idxs[i + 1] is None:
                break
            # If first_style_idx == begin and next_style_idx == end
            if (
                self.style_idxs[i] == self.begin_fr_idx
                and self.style_idxs[i + 1] == self.end_fr_idx
            ):
                print("First last hit")
                sequences.append(
                    Sequence(
                        self.begin_fr_idx,
                        self.end_fr_idx,
                        self.style_paths[i],
                        self.style_paths[i + 1],
                    )
                )

            # If the first style index is the first frame in the sequence
            elif (
                self.style_idxs[i] == self.begin_fr_idx
                and self.style_idxs[i + 1] != self.end_fr_idx
            ):
                print("First off last hit")
                sequences.append(
                    Sequence(
                        self.begin_fr_idx,
                        self.style_idxs[i + 1],
                        self.style_paths[i],
                        self.style_paths[i + 1],
                        is_all=False,
                        is_blend=True
                    )
                )
                if i + 2 < self.num_style_frs:
                    sequences.append(
                        Sequence(
                            begin_fr_idx=self.style_idxs[i + 1],
                            end_fr_idx=self.style_idxs[i + 2],
                            style_start_fr=self.style_paths[i + 1],
                            style_end_fr=self.style_paths[i + 2],
                            is_all=False,
                            is_blend=True
                        )
                    )
                else:
                    sequences.append(
                        Sequence(
                            begin_fr_idx=self.style_idxs[i + 1],
                            end_fr_idx=self.end_fr_idx,
                            style_start_fr=self.style_paths[i + 1],
                            is_all=False,
                            is_blend=False
                        )
                    )

            # If the second style index is the last frame in the sequence
            elif (
                self.style_idxs[i] != self.begin_fr_idx
                and self.style_idxs[i + 1] == self.end_fr_idx
            ):
                sequences.append(
                    Sequence(
                        self.style_idxs[i],
                        self.end_fr_idx,
                        self.style_paths[i],
                        self.style_paths[i + 1],
                    )
                )

            elif (
                self.style_idxs[i] != self.begin_fr_idx
                and self.style_idxs[i + 1] != self.end_fr_idx
                and self.style_idxs[i] in self.img_idxs
                and self.style_idxs[i + 1] in self.img_idxs
            ):
                sequences.append(
                    Sequence(
                        self.style_idxs[i],
                        self.style_idxs[i + 1],
                        self.style_paths[i],
                        self.style_paths[i + 1],
                    )
                )

        return sequences

    @staticmethod
    def generate_subsequences(sequence: Sequence, chunk_size=10, overlap_frs=1):
        print("I AM RUN!")
        subsequences = []
        init_idx = sequence.begin_fr_idx
        final_idx = sequence.end_fr_idx
        begin_fr_idx = init_idx
        last_end_fr_idx = -1  # Keep track of the last endFrame

        while begin_fr_idx < final_idx:
            end_fr_idx = min(begin_fr_idx + chunk_size + overlap_frs, final_idx)

            # If remaining frames are less than chunk size and already covered by previous subsequence
            if (final_idx - begin_fr_idx < chunk_size) and (
                last_end_fr_idx >= final_idx
            ):
                break

            # Create a subsequence
            subseq = Subsequence(
                init_idx,
                final_idx,
                begin_fr_idx,
                end_fr_idx,
                sequence.style_start_fr,
                sequence.style_end_fr,
            )
            subsequences.append(subseq)
            last_end_fr_idx = end_fr_idx  # Update last_endFrame

            # Update begFrame for the next iteration
            new_begin_fr_idx = end_fr_idx - overlap_frs

            # Check to prevent infinite loop
            if new_begin_fr_idx <= begin_fr_idx:
                break

            begin_fr_idx = new_begin_fr_idx

        return subsequences

    def __call__(self, chunk_size=10, overlap=1):
        sequences = self._set_sequence()
        subsequences = []
        for sequence in sequences:
            subsequences += self.generate_subsequences(sequence, chunk_size, overlap)
        return subsequences
