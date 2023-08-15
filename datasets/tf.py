from .hsi import HsiDataset


class TF(HsiDataset):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(root)
