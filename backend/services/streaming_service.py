import asyncio
import json
from typing import AsyncGenerator

from backend.core.state import progress


async def sse_generator() -> AsyncGenerator[str, None]:
    idx = 0
    yield "retry: 1000\n\n"
    while True:
        with progress.lock:
            nuevos = progress.events[idx:]
            done   = progress.is_complete

        for event in nuevos:
            idx += 1
            yield f"data: {json.dumps(event)}\n\n"

        if done and idx >= len(progress.events):
            yield f"data: {json.dumps({'tipo': 'fin_stream'})}\n\n"
            break

        await asyncio.sleep(0.15)
