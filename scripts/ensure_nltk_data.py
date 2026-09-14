#!/usr/bin/env python3

import argparse
import ssl
from pathlib import Path

import nltk

RESOURCE_PATHS = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        default="environment/playpen/nltk_data",
        help="Directory where NLTK resources should be stored.",
    )
    parser.add_argument(
        "resources",
        nargs="*",
        default=["stopwords", "wordnet"],
        help="NLTK resources to ensure are available.",
    )
    args = parser.parse_args()

    download_dir = Path(args.download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(download_dir))

    for resource in args.resources:
        resource_path = RESOURCE_PATHS.get(resource, resource)
        try:
            nltk.data.find(resource_path)
            print(f"Already available: {resource} ({download_dir})")
        except LookupError:
            print(f"Downloading: {resource} -> {download_dir}")
            ok = False
            try:
                ok = nltk.download(resource, download_dir=str(download_dir), quiet=True)
            except Exception:
                ok = False
            if not ok:
                ssl._create_default_https_context = ssl._create_unverified_context
                ok = nltk.download(resource, download_dir=str(download_dir), quiet=True)
            if not ok:
                raise SystemExit(f"Failed to download NLTK resource: {resource}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
