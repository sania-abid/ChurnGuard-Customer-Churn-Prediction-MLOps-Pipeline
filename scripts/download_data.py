from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_app.config import get_settings
from churn_app.data import download_dataset


def main() -> None:
    settings = get_settings()
    path = download_dataset(settings.dataset_url, settings.raw_data_path)
    print(f"Downloaded dataset to {path}")


if __name__ == "__main__":
    main()
