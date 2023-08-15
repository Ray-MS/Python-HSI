from .hsi import HsiDataset


class IP(HsiDataset):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(root)

