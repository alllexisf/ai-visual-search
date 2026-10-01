import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / '.env')


def integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, default))
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv('API_KEY', ''))
    frontend_origin: str = field(default_factory=lambda: os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173'))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv('INDEX_DIR', str(BACKEND / 'data/index'))))
    dataset_dir: Path = field(default_factory=lambda: Path(os.getenv('DATASET_DIR', str(BACKEND / 'data/datasets'))))
    model_id: str = field(default_factory=lambda: os.getenv('CLIP_MODEL_ID', 'openai/clip-vit-base-patch32'))
    model_revision: str = field(default_factory=lambda: os.getenv('CLIP_MODEL_REVISION', '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'))
    max_upload_bytes: int = field(default_factory=lambda: integer('MAX_UPLOAD_MB', 10) * 1024 * 1024)
    max_image_pixels: int = field(default_factory=lambda: integer('MAX_IMAGE_PIXELS', 25_000_000))
    max_top_k: int = field(default_factory=lambda: integer('MAX_TOP_K', 50))
    rate_limit: int = field(default_factory=lambda: integer('SEARCH_RATE_LIMIT', 30))
    asset_rate_limit: int = field(default_factory=lambda: integer('ASSET_RATE_LIMIT', 600))
    rate_window: int = field(default_factory=lambda: integer('RATE_LIMIT_WINDOW_SECONDS', 60))
    torch_threads: int = field(default_factory=lambda: integer('TORCH_THREADS', min(4, os.cpu_count() or 1)))

    def validate(self) -> None:
        if not self.api_key or self.api_key == 'replace-with-a-random-local-key':
            raise ValueError('Set a nonempty API_KEY in backend/.env before starting.')
        if self.frontend_origin == '*' or not self.frontend_origin.startswith(('http://', 'https://')):
            raise ValueError('FRONTEND_ORIGIN must be one explicit HTTP(S) origin')
        if self.max_top_k > 500:
            raise ValueError('MAX_TOP_K must be <= 500')
