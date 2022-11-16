import argparse
from pathlib import Path

import whisper
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--models",
    nargs="+",
    required=True,
    default=[],
    help="Models to download, e.g. `tiny, small, medium`",
)

parser.add_argument("--target", default="~/.cache/whisper", help="Target directory")


def main():
    download_path = str(Path(args.target).expanduser())

    logger.info(f"Downloading whisper models to {download_path}")

    logger.info(f"Downloading models: {args.models}")

    for m in args.models:
        url = whisper._MODELS[m]
        logger.info(f"Downloading model {m} from {url} to {download_path}")
        whisper._download(url=url, root=download_path, in_memory=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
