# configs/config.py

"""
Configuration management for KYC service.
Simple, efficient, production-ready.
"""

import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_yaml_config() -> Dict[str, Any]:
    """Load defaults.yaml configuration."""
    config_path = Path(__file__).parent / "defaults.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings from .env file."""
    
    # Environment
    ENV: str = Field(default="development", description="Environment: development|staging|production")
    
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="DEBUG|INFO|WARNING|ERROR")
    
    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated allowed origins"
    )
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = Field(default=10, description="Max file size in MB")
    
    # Models
    USE_GPU: bool = Field(default=False, description="Use GPU for model inference")
    
    # Storage (if using S3)
    S3_BUCKET: Optional[str] = Field(default=None)
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    
    # API Security (optional)
    API_KEY: Optional[str] = Field(default=None, description="API key for authentication")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class Config:
    """
    Centralized configuration manager.
    Merges defaults.yaml with environment variables.
    """
    
    def __init__(self):
        # Load YAML defaults
        self._defaults = load_yaml_config()
        
        # Load environment overrides
        self.env = EnvironmentSettings()
        
        # Merge configurations
        self._config = self._merge_configs()
        
        logger.info(f"Configuration loaded for environment: {self.env.ENV}")
    
    def _merge_configs(self) -> Dict[str, Any]:
        """Merge YAML defaults with environment variables."""
        merged = self._defaults.copy()
        
        # Override with environment variables
        if self.env.LOG_LEVEL:
            merged["project"]["log_level"] = self.env.LOG_LEVEL
        
        if self.env.HOST:
            merged["server"]["host"] = self.env.HOST
        
        if self.env.PORT:
            merged["server"]["port"] = self.env.PORT
        
        if self.env.S3_BUCKET:
            merged["storage"]["s3_bucket"] = self.env.S3_BUCKET
        
        # Add computed values
        merged["project"]["env"] = self.env.ENV
        merged["api"] = {
            "cors_origins": self.env.CORS_ORIGINS.split(","),
            "max_upload_size": self.env.MAX_UPLOAD_SIZE_MB * 1024 * 1024,  # Convert to bytes
            "use_gpu": self.env.USE_GPU,
            "api_key": self.env.API_KEY
        }
        
        return merged
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely access nested config values.
        
        Example:
            config.get("models", "face_detection", "url")
            config.get("server", "port", default=8000)
        """
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value
    
    @property
    def all(self) -> Dict[str, Any]:
        """Return full configuration dict."""
        return self._config
    
    @property
    def server_host(self) -> str:
        """Shortcut for server host."""
        return self.get("server", "host", default="0.0.0.0")
    
    @property
    def server_port(self) -> int:
        """Shortcut for server port."""
        return self.get("server", "port", default=8000)
    
    @property
    def log_level(self) -> str:
        """Shortcut for log level."""
        return self.get("project", "log_level", default="INFO")
    
    @property
    def cors_origins(self) -> list:
        """Shortcut for CORS origins."""
        return self.get("api", "cors_origins", default=["*"])
    
    @property
    def max_upload_size(self) -> int:
        """Shortcut for max upload size in bytes."""
        return self.get("api", "max_upload_size", default=10485760)  # 10MB
    
    @property
    def use_gpu(self) -> bool:
        """Shortcut for GPU usage."""
        return self.get("api", "use_gpu", default=False)


# Global singleton instance
config = Config()