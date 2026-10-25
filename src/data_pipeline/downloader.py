"""Download NHANES XPT files for cycles H–J (2013–2018)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

from src.config import NHANES_CYCLES

logger = logging.getLogger(__name__)

CDC_DATAFILES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# cycle folder key -> (public data year, file suffix letter)
_CYCLE_URL: dict[str, tuple[int, str]] = {
    "2013_2014": (2013, "H"),
    "2015_2016": (2015, "I"),
    "2017_2018": (2017, "J"),
}

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

_XPORT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"

MEC_FILE_STEMS = ("DEMO", "BMX", "BPX", "GHB", "TCHOL", "HDL", "DIQ", "RXQ_RX")
FASTING_FILE_STEMS = ("GLU", "TRIGLY", "FASTQX")


def _xpt_filename(stem: str, suffix: str) -> str:
    if stem == "RXQ_RX":
        return f"RXQ_RX_{suffix}.XPT"
    return f"{stem}_{suffix}.XPT"


def file_url(cycle_key: str, stem: str) -> str:
    year, letter = _CYCLE_URL[cycle_key]
    name = _xpt_filename(stem, letter)
    return f"{CDC_DATAFILES_BASE}/{year}/DataFiles/{name}"


def is_valid_xport(path: Path) -> bool:
    """True if the file looks like a SAS XPORT dataset (not an HTML error page)."""
    try:
        with open(path, "rb") as f:
            head = f.read(256)
    except OSError:
        return False
    if head.startswith(b"<!") or head.startswith(b"<html"):
        return False
    return _XPORT_MAGIC in head[:80]


def download_file(url: str, dest: Path, session: requests.Session | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    resp = sess.get(url, timeout=120, stream=True, allow_redirects=True)
    resp.raise_for_status()
    content_type = (resp.headers.get("content-type") or "").lower()
    if "html" in content_type:
        raise ValueError(f"expected XPT binary, got HTML from {url}")
    total = int(resp.headers.get("content-length", 0))
    hasher = hashlib.sha256()
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                hasher.update(chunk)
                bar.update(len(chunk))
    digest_path = dest.with_suffix(dest.suffix + ".sha256")
    digest_path.write_text(hasher.hexdigest() + "\n", encoding="utf-8")
    if not is_valid_xport(dest):
        dest.unlink(missing_ok=True)
        digest_path.unlink(missing_ok=True)
        raise ValueError(f"downloaded file is not a valid SAS XPORT file: {url}")
    return dest


def verify_sha256(path: Path) -> bool:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not digest_path.exists():
        return False
    expected = digest_path.read_text(encoding="utf-8").strip().split()[0]
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected


def download_cycle(
    cycle_key: str,
    raw_root: Path,
    stems: Iterable[str] = MEC_FILE_STEMS + FASTING_FILE_STEMS,
    skip_existing: bool = True,
) -> list[Path]:
    if cycle_key not in _CYCLE_URL:
        raise ValueError(f"unknown cycle {cycle_key}")
    dest_dir = raw_root / cycle_key
    paths: list[Path] = []
    with requests.Session() as session:
        session.headers.update(_DEFAULT_HEADERS)
        for stem in stems:
            url = file_url(cycle_key, stem)
            dest = dest_dir / Path(url).name
            if skip_existing and dest.exists() and is_valid_xport(dest) and verify_sha256(dest):
                logger.info("skip %s (unchanged)", dest.name)
                paths.append(dest)
                continue
            logger.info("downloading %s", url)
            paths.append(download_file(url, dest, session=session))
    return paths


def download_all_cycles(raw_root: Path | str = "data/raw", skip_existing: bool = True) -> None:
    root = Path(raw_root)
    for cycle in NHANES_CYCLES:
        download_cycle(cycle, root, skip_existing=skip_existing)


def purge_invalid_xpt(raw_root: Path | str = "data/raw") -> int:
    """Remove .XPT files that are not valid SAS transport (e.g. HTML error pages)."""
    root = Path(raw_root)
    removed = 0
    for path in root.rglob("*.XPT"):
        if not is_valid_xport(path):
            logger.warning("removing invalid XPT: %s", path)
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
            removed += 1
    return removed
