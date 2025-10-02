import os
import yaml
from pathlib import Path
from pydantic import BaseSettings, Field
from typing import Any, Dict

def load_yaml_defaults() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "defaults.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class EnvSettings(BaseSettings):

    ENV: str = Field(default="local")  # local/staging/prod
    S3_BUCKET: str | None = None
    DATABASE_URL: str | None = None
    LOG_LEVEL: str | None = None

    class Config:
        env_file = ".env"  
        env_file_encoding = "utf-8"
        case_sensitive = True

class Config:
    def __init__(self):
        
        self.defaults = load_yaml_defaults()

        self.env = EnvSettings()

        self._config = self.merge_configs()

    def merge_configs(self) -> Dict[str, Any]:
        merged = self.defaults.copy()

        # override: log level
        if self.env.LOG_LEVEL:
            merged["project"]["log_level"] = self.env.LOG_LEVEL

        # override: s3 bucket
        if self.env.S3_BUCKET:
            merged["storage"]["s3_bucket"] = self.env.S3_BUCKET

        # env-specific stuff
        merged["project"]["env"] = self.env.ENV

        return merged

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Access nested config values safely.
        Example: config.get("models", "face_detection", "url")
        """
        cfg = self._config
        for key in keys:
            if cfg is None:
                return default
            cfg = cfg.get(key)
        return cfg if cfg is not None else default

    @property
    def all(self) -> Dict[str, Any]:
        return self._config

config = Config()
