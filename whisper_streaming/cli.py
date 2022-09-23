#!/usr/bin/env python3

import argparse
import queue
from logging import INFO, getLogger
from typing import Optional, Union

import sounddevice as sd
import torch
from whisper import available_models
from whisper.audio import N_FRAMES, SAMPLE_RATE
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from whisper_streaming.schema import WhisperConfig
from whisper_streaming.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)


def transcribe_from_mic(
    *,
    config: WhisperConfig,
    sd_device: Optional[Union[int, str]],
) -> None:
    wsp = WhisperStreamingTranscriber(config=config)
    q = queue.Queue()

    def sd_callback(indata, frames, time, status):
        if status:
            logger.warning(status)
        q.put(indata.ravel())

    logger.info("Ready to transcribe")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=N_FRAMES * 10,  # FIXME
        device=sd_device,
        dtype="float32",
        channels=1,
        callback=sd_callback,
    ):
        while True:
            segment = q.get()
            for chunk in wsp.transcribe(segment=segment):
                print(f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}")


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=available_models(),
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for PyTorch inference",
    )
    parser.add_argument(
        "--beam_size",
        "-b",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--mic",
    )

    return parser.parse_args()


def main() -> None:
    opts = get_opts()
    logger.setLevel(INFO)
    if opts.beam_size <= 0:
        opts.beam_size = None
    try:
        opts.mic = int(opts.mic)
    except Exception:
        pass

    config = WhisperConfig(
        model_name=opts.model,
        language=opts.language,
        device=opts.device,
        beam_size=opts.beam_size,
    )
    transcribe_from_mic(
        config=config,
        sd_device=opts.mic,
    )


if __name__ == "__main__":
    main()