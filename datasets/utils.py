import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

USER_AGENT = "pytorch/vision"


def _save_response_content(
        content: Iterator[bytes],
        destination: Path,
        length: Optional[int] = None,
) -> None:
    with destination.open("wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def download_url(url: str, filename: Path, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(iter(lambda: response.read(chunk_size), b""), filename, length=response.length)
