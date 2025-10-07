<div align="center">

# kycflow

Lightweight, production-ready KYC verification pipeline combining face detection, face matching, and OCR using Yunet, InsightFace, and EasyOCR. Ships with a clean orchestration layer, configurable settings, a CLI for local runs, and tests.

</div>

---

### Key Features
- **End-to-end KYC pipeline**: ID face detection, selfie face detection, face verification, and OCR extraction
- **Fast and reliable**: ONNX Yunet for face detection, InsightFace for embeddings, EasyOCR for text
- **Configurable**: YAML defaults with environment overrides via `.env`
- **Safe-by-default**: Clear error handling, structured results compatible with frontend workflows
- **Batteries included**: CLI runner, logs, model download helper, and tests

### Tech Stack
- **Python**: 3.10 – 3.13
- **Core libs**: OpenCV, InsightFace, EasyOCR, ONNX Runtime, Torch
- **Config**: Pydantic Settings + YAML
- **Dev**: Poetry, Pytest, Black, isort

## Project Structure
```text
kycflow/
├─ app/
│  ├─ main.py                 # Orchestration: loads services, runs pipeline, CLI entry
│  └─ services/
│     ├─ face_detector_id.py  # Face detection on ID/selfie (Yunet ONNX)
│     ├─ face_matcher.py      # InsightFace based face similarity/verification
│     ├─ ocr_extractor.py     # EasyOCR extraction with structured result
│     
├─ api/
│  ├─ api.py                  # FastAPI scaffold (extend to expose HTTP API)
│  └─ schemas.py              # Pydantic models for request/response
├─ configs/
│  ├─ config.py               # Config manager merging YAML + environment
│  └─ defaults.yaml           # Default configuration values
├─ models/
│  └─ yunet.onnx              # Face detector model (downloaded or provided)
├─ scripts/
│  └─ download_models.py      # Helper script to fetch required models
├─ data/
│  ├─ id/                     # Sample ID images
│  └─ realtime_photo/         # Sample selfie images
├─ tests/                     # Pytest suite and sample outputs
├─ utils/                     # Helpers
│  └─ logger.py               # Logger setup/utilities
├─ logs/                      # Runtime logs
├─ pyproject.toml             # Poetry project + dependencies
├─ README.md                  # You are here
└─ Makefile                   # Optional convenience targets
```

## Installation
1) Ensure Python is installed (3.10 – 3.13). Prefer a virtual environment.

2) Install dependencies with Poetry:
```bash
pip install --upgrade pip
pip install poetry
poetry install --no-root
```

3) Download models (if not already present):
```bash
poetry run python -m scripts.download_models
```

4) Optional: Create a `.env` in the project root to override settings (see Configuration).

## Configuration
Configuration merges `configs/defaults.yaml` with environment variables via `.env`.

Environment variables supported (subset):
- `ENV`: development|staging|production (default: development)
- `HOST`: server host (default: 0.0.0.0)
- `PORT`: server port (default: 8000)
- `LOG_LEVEL`: DEBUG|INFO|WARNING|ERROR (default: INFO)
- `CORS_ORIGINS`: comma-separated origins (default: http://localhost:3000,http://localhost:5173)
- `MAX_UPLOAD_SIZE_MB`: max upload size in MB (default: 10)
- `USE_GPU`: true|false to enable GPU inference where supported (default: false)
- `API_KEY`: optional API key if you add auth to the HTTP API

Where they are consumed:
- `configs/config.py` centralizes loading and provides helpers like `config.get(...)`, `config.server_host`, `config.use_gpu`, etc.

## Usage
### CLI (Local Processing)
The orchestration entrypoint is `app/main.py`. Example:
```bash
poetry run python -m app.main \
  --id data/id/sample_id.jpg \
  --selfie data/realtime_photo/sample_selfie.jpg \
  --output result.json
```
Flags:
- `--id`: path to the ID document image
- `--selfie`: path to the selfie image
- `--no-ocr`: skip OCR extraction
- `--output`: optional JSON file to write the full result

Output structure (high level):
- `verification_status`: approved|rejected|pending
- `confidence_score`: overall confidence (0–1)
- `face_match_score`: similarity score (0–1)
- `ocr_data`: extracted text and fields (if OCR enabled)
- `processing_time_ms`, `timestamp`, and `details` including detection and similarity metrics

### HTTP API (Scaffold)
`api/api.py` is provided as a starting point for a FastAPI service. Extend it to expose an endpoint that:
- Accepts an ID image and a selfie (multipart or URLs)
- Invokes `app.main.get_kyc_processor().process_kyc(...)`
- Returns the pipeline result

Once implemented, you can run with:
```bash
poetry run uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

## Models
- Face detection: `models/yunet.onnx`
- Face matching: InsightFace will manage backbone weights (downloaded on first run or via helper)
- OCR: EasyOCR downloads language packs on first use

Use `scripts/download_models.py` to prefetch models where supported. Ensure the `models/` folder exists and is writable.

## Logging
- Application logs at `logs/app.log` and download logs at `logs/download.log`
- The CLI run also prints progress and summary to stdout
- Adjust log level using `LOG_LEVEL` in `.env`

## Testing
Run the test suite with Pytest:
```bash
poetry run pytest -q
```
Artifacts from sample runs are available in `tests/outputs/` for quick inspection.

## Development
- Code style: Black and isort are configured in `pyproject.toml`
- Minimum Python: `>=3.10,<3.14`
- Dependency management: Poetry (`pyproject.toml`)
- Prefer small, focused modules; keep orchestration in `app/main.py`

Suggested workflow:
- Add/iterate services in `app/services/`
- Access config via `from configs.config import config`
- Write unit tests under `tests/`

## Performance Notes
- CPU-only works out-of-the-box; set `USE_GPU=true` to enable GPU where supported
- For throughput, batch runs or parallelize at the orchestration level (already uses threads for sub-steps)
- Tune `verification.similarity_threshold` and `ocr.confidence_threshold` in `configs/defaults.yaml`

## Security and Compliance
- If exposing an HTTP API, consider enabling API-key headers and CORS restrictions
- Avoid persisting PII unless necessary; scrub logs if storing traces
- Validate image sizes against `MAX_UPLOAD_SIZE_MB`

## Troubleshooting
- Model not found: re-run `scripts/download_models.py` and verify `models/yunet.onnx`
- Import errors: ensure `poetry install` completed and you are in the Poetry shell or using `poetry run`
- EasyOCR language data: first run may download; ensure network access
- Low match confidence: verify face crops are frontal and images are sharp; adjust thresholds

## License
Specify your license in this file or in `pyproject.toml` metadata.

## Acknowledgements
- Yunet ONNX face detector (OpenCV)
- InsightFace for face recognition
- EasyOCR for text extraction

---

Happy verifying!