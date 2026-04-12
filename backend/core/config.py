from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

executor = ThreadPoolExecutor(max_workers=1)
