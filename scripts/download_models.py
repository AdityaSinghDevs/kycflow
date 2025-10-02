import os 
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib
import yaml 

from utils.logger import get_logger
from configs.config import config

logger = get_logger(name=__name__, log_file='download.log')

def download_file(url : str, dest : Path, chunk_size: int = 1024)->None:
    '''
    Downloads a file from a URL with tqdm bar
    skip if already exists
    '''
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info(f"[SKIP :] {dest.name} already exists at {dest}.")
        return
    
    logger.info(f"[DOWNLOAD :] {dest.name} from {url} ...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        logger.info(f"[DONE] Downloaded {dest.name} to {dest}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to download {dest.name}: {e}")
        raise e

def handle_model(model_cfg: dict, models_dir: Path) -> None:
    """
    Process a single model: pip install (log) or download URL.
    """
    name = model_cfg.get("name")
    source = model_cfg.get("source")
    local_file = model_cfg.get("local_file")
    url = model_cfg.get("url")
    pip_package = model_cfg.get("pip_package")

    if source == "pip":
        logger.info(f"[PIP] Model '{name}' installed via Poetry: {pip_package}")
        return

    if source == "url" and url and local_file:
        dest_path = models_dir / local_file
        download_file(url, dest_path)
    else:
        logger.warning(f"[SKIP] Model '{name}' has no valid URL or pip info.")

def main():
    models_dir = Path(config.get("paths", "models_dir", default="models"))
    models_cfg = config.get("models", default={})

    if not models_cfg:
        logger.warning("[WARN] No models found in configuration.")
        return

    logger.info(f"[INFO] Starting model downloads to {models_dir.resolve()}")
    
    for key, model in models_cfg.items():
        try:
            handle_model(model, models_dir)
        except Exception as e:
            logger.error(f"[ERROR] Failed to process model '{key}': {e}")

    logger.info("[SUCCESS] All models processed.")


if __name__ == "__main__":
    main()
