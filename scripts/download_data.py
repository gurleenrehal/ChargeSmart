"""Download Caltech ACN-Data sessions (official API) or fall back to the synthetic sample.

Usage:
    python scripts/download_data.py --token YOUR_TOKEN --site caltech --start 2019-01-01 --end 2019-12-31
    python scripts/download_data.py --sample            # (re)generate the synthetic, ACN-schema sample only

The token can also come from the ACN_API_TOKEN environment variable or a .env file.
Register for a free token at https://ev.caltech.edu/register.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import data_loader


def _read_dotenv():
    env = config.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    _read_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=os.environ.get("ACN_API_TOKEN"))
    ap.add_argument("--site", default=os.environ.get("ACN_SITE", "caltech"), choices=["caltech", "jpl", "office001"])
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--sample", action="store_true", help="only (re)generate the synthetic sample")
    args = ap.parse_args()

    if args.sample or not args.token or args.token == "your_token_here":
        if not args.sample:
            print("No ACN API token provided -> using the synthetic sample dataset (NOT real data).")
        df = data_loader.generate_synthetic_sample()
        print(f"Wrote {len(df)} synthetic sessions to {config.SAMPLE_CSV}")
        return 0
    try:
        df = data_loader.download_acn_data(args.token, args.site, datetime.fromisoformat(args.start),
                                           datetime.fromisoformat(args.end))
    except RuntimeError as exc:
        print(f"Download failed: {exc}\nFalling back to the synthetic sample.", file=sys.stderr)
        if not config.SAMPLE_CSV.exists():
            data_loader.generate_synthetic_sample()
        return 1
    print(f"Downloaded {len(df)} real ACN-Data sessions to {config.RAW_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
