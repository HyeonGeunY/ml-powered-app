from urllib.request import urlretrieve
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import os
import socket
import py7zr

socket.setdefaulttimeout(30)

class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting writers.stackexchage data")
    curdir = os.getcwd()
    os.chdir(dirname)

    with py7zr.SevenZipFile(filename, mode='r') as z:
        z.extractall()
        
    os.chdir(curdir)

def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    _download_url(metadata["url"], filename)
    return filename

def _download_url(url, filename):

    try:
        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
            urlretrieve(url, filename, reporthook=t.update_to, data=None)
    
    except socket.timeout:
        count = 1

        while count <= 5:
            try:
                with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
                    urlretrieve(url, filename, reporthook=t.update_to, data=None)
                break
            except socket.timeout:
                err_info = f"Reloading for {count} times"
                print(err_info)
                count += 1
        
        if count > 5:
            print("downloading failed")