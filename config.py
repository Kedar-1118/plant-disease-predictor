from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AppConfig:
    upload_folder: Path
    max_content_length: int
    allowed_extensions: tuple[str, ...]
    debug: bool
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        upload_folder = BASE_DIR / os.getenv("UPLOAD_FOLDER", "static/uploads")
        max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
        allowed_extensions = tuple(
            ext.strip().lower()
            for ext in os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,webp,bmp").split(",")
            if ext.strip()
        )
        debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", "5000"))

        return cls(
            upload_folder=upload_folder,
            max_content_length=max_content_length,
            allowed_extensions=allowed_extensions,
            debug=debug,
            host=host,
            port=port,
        )
