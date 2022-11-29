import asyncio
import base64
import json
from logging import getLogger
from typing import Any
from typing import Dict
from typing import Final, Optional
from typing import Union

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedOK
from whispering.schema import ParsedChunk

from whispering.schema import CURRENT_PROTOCOL_VERSION, Context
from whispering.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)

MIN_PROTOCOL_VERSION: Final[int] = int("000_006_000")
MAX_PROTOCOL_VERSION: Final[int] = CURRENT_PROTOCOL_VERSION


def deserialize_message(message: Union[str, bytes]) -> Dict[str, Any]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")

    return json.loads(message)


hard_coded_context_vars = {
    # "compression_ratio_threshold": 1.2,
    # "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}


async def serve_with_websocket_main(websocket):
    global g_wsp
    idx: int = 0
    ctx: Optional[Context] = None

    while True:

        try:
            message = await websocket.recv()
        except ConnectionClosedOK:
            break

        force_padding = False

        speaker = None

        try:
            message = deserialize_message(message)
        except Exception as e:
            logger.exception(f"Failed to deserialize message: {e}")
            continue

        logger.debug(
            f"Audio #: {idx} -- "
            f"message type={type(message)} and `bool(ctx)`={bool(ctx)}"
        )

        if "context" in message and not ctx:
            v = message["context"]
            logger.debug(f"Got context for bot ID {message.get('bot_id')}: {v}")
            if v is not None:
                logger.debug(
                    f"Updating received context with hard-coded values: "
                    f"{hard_coded_context_vars}"
                )
                v.update(hard_coded_context_vars)
                ctx = Context.parse_obj(v)
            else:
                await websocket.send(json.dumps({"error": "unsupported message"}))
                return

            if ctx.protocol_version < MIN_PROTOCOL_VERSION:
                await websocket.send(
                    json.dumps(
                        {
                            "error": f"protocol_version is older than {MIN_PROTOCOL_VERSION}"
                        }
                    )
                )
            elif ctx.protocol_version > MAX_PROTOCOL_VERSION:
                await websocket.send(
                    json.dumps(
                        {
                            "error": f"protocol_version is newer than {MAX_PROTOCOL_VERSION}"
                        }
                    )
                )
                return

            continue

        if ctx is None:
            await websocket.send(json.dumps({"error": "no context"}))
            return

        speaker = message.get("speaker")
        bot_id = message.get("bot_id")
        logger.info(f"Speaker: {speaker}")
        if message.get("begin_new_speaker"):
            logger.warning(f"Received `begin_new_speaker` for bot ID {bot_id}")
            force_padding = True

        audio_bytes = base64.b64decode(message.get("b64_encoded_audio", ""))
        logger.debug(
            f"Processing audio of length {len(audio_bytes)} for bot ID {bot_id}"
        )
        audio = np.frombuffer(audio_bytes, dtype=np.dtype(ctx.data_type)).astype(
            np.float32
        )

        for chunk in g_wsp.transcribe(
            audio=audio,
            ctx=ctx,
            speaker=speaker,
            bot_id=bot_id,
            force_padding=force_padding,  # type: ignore
        ):
            if chunk:
                chunk: ParsedChunk
                logger.debug(f"Returning chunk for bot ID {bot_id}: {chunk.json()}")
                await websocket.send(chunk.json())
        #
        # if force_padding:
        #     await websocket.send(json.dumps({"close_connection": True}))

        idx += 1


async def serve_with_websocket(
    *,
    wsp: WhisperStreamingTranscriber,
    host: str,
    port: int,
):
    logger.info(f"Serve at {host}:{port}")
    logger.info("Make secure with your responsibility!")
    global g_wsp
    g_wsp = wsp

    try:
        async with websockets.serve(  # type: ignore
            serve_with_websocket_main,
            host=host,
            port=port,
            max_size=999999999,
        ):
            await asyncio.Future()
    except KeyboardInterrupt:
        pass
