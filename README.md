# Setup

## 1. Prerequisites
- `Python 3.12` 
- `uv` this project is using uv from astral https://docs.astral.sh/uv/getting-started/installation/


## 2. Install Dependencies
- `uv sync` - this is all that is needed to download uv Dependencies

## 3. GPU 
The script automatically uses `mps` if available, otherwise it will run on `cpu`

## 4. Run Script
```bash
uv run model.py --save FinalRun
```
first run will download dataset and --save flag will save an image to the results

