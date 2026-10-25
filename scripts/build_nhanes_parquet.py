"""CLI: download NHANES (optional) and build pooled Parquet tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_pipeline.downloader import download_all_cycles, purge_invalid_xpt
from src.data_pipeline.harmonizer import export_processed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Build NHANES pooled Parquet cohorts")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument(
        "--purge-invalid",
        action="store_true",
        help="Delete raw .XPT files that are not valid SAS transport before downloading",
    )
    args = p.parse_args()
    if args.purge_invalid:
        n = purge_invalid_xpt(args.raw)
        logging.info("purged %s invalid XPT file(s)", n)
    if not args.skip_download:
        download_all_cycles(args.raw)
    mec_path, fast_path = export_processed(args.processed, args.raw)
    print(f"Wrote {mec_path} and {fast_path}")


if __name__ == "__main__":
    main()
