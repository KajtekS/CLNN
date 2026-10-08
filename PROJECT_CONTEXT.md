# CLNN Project Context

## 1. Project goal

This repository is a Python-based research/prototype project for a CLNN pipeline built around a medallion-style data flow:

- raw
- bronze
- silver
- gold

The project is organized as a pipeline-oriented skeleton for processing video data and extracting relevant signals for model training or inference. Based on the current codebase, the intended workflow is:

1. load data from a configured source,
2. run face detection / preprocessing,
3. normalize and resample data,
4. pass processed data into a higher-level model stage.

The code currently looks like an early-stage scaffolding implementation rather than a fully finished product. Several modules define abstract interfaces and placeholder methods, which is typical for a research prototype still being assembled.

---

## 2. Repository structure

This project is organized as a compact Python research prototype with a clear stage-based architecture.

```text
CLNN/
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
├── CONFIGS/
│   └── HOW_CONFIG_SHOOULD_LOOK_LIKE.yaml
├── DATA/
│   ├── BRONZE/
│   │   └── .gitkeep
│   ├── GOLD/
│   │   └── .gitkeep
│   ├── RAW/
│   │   └── .gitkeep
│   └── SILVER/
│       └── .gitkeep
├── LOADER/
│   └── loader.py
├── MODELS/
│   └── model.py
├── PIPELINE/
│   └── pipeline.py
├── PROCESING/
│   ├── .DS_Store
│   └── preproc.py
├── TESTS/
│   └── test_tools.py
├── TOOLS/
│   ├── __init__.py
│   ├── config_parser.py
│   ├── resampling.py
│   └── yolov8/
│       └── yolov8n-face.pt
└── .DS_Store
```

Project-level summary:

- `main.py` is the entry point.
- `CONFIGS/` contains YAML config examples.
- `DATA/` stores the medallion-style data stages.
- `LOADER/`, `PIPELINE/`, `MODELS/`, and `PROCESING/` define the processing architecture.
- `TOOLS/` holds config, helper utilities, and the YOLO face model weights.
- `TESTS/` covers the utility-level validation.

---

## 3. How to work with this project as a developer

If you are coding in this repository, the usual workflow is:

1. Read the config contract in `CONFIGS/HOW_CONFIG_SHOOULD_LOOK_LIKE.yaml`.
2. Update the runtime behavior in the relevant layer:
   - dataset loading in `LOADER/loader.py`
   - orchestration in `PIPELINE/pipeline.py`
   - preprocessing in `PROCESING/preproc.py`
   - model logic in `MODELS/model.py`
3. Add or fix tests in `TESTS/test_tools.py` for the behavior you changed.
4. Run the focused validation command:

```bash
pytest
```

or a single test file:

```bash
pytest TESTS/test_tools.py
```

When changing the project, keep the architecture clear:

- configuration belongs in `CONFIGS/`
- actual reusable logic belongs in `TOOLS/`
- processing steps belong in `PROCESING/` and `PIPELINE/`
- higher-level algorithmic behavior belongs in `MODELS/`
- tests belong in `TESTS/`

This keeps the project modular and compatible with the medallion pipeline idea.

---

## 4. Directory-by-directory explanation

### `CONFIGS/`

Purpose:
- contains sample YAML config definitions that describe how the pipeline should run.

Key file:
- `HOW_CONFIG_SHOOULD_LOOK_LIKE.yaml`

This is not a runtime-generated config; it acts as a schema/example. It shows the expected structure for:

- `MAIN.PROCESS_LAYER`
- `RAW.PATH`
- `BRONZE.PATH`, sample sampling settings
- `SILVER.PATH`, processing parameters
- `GOLD.PATH`, chunking and axis ordering
- `MODEL.GPU`, `MODEL.NAME`

This is the main way the project is supposed to be configured.

---

### `DATA/`

Purpose:
- storage location for different data stages.

Subfolders:
- `RAW/`
- `BRONZE/`
- `SILVER/`
- `GOLD/`

Each folder is currently mostly empty (using `.gitkeep` placeholders). This indicates the project envisions a medallion workflow in which data moves through layers as it is cleaned, transformed, and prepared for modeling.

---

### `LOADER/`

Purpose:
- defines abstractions for loading data.

Key file:
- `loader.py`

This module defines an abstract base class called `loader` with a `load_data` method. In the current state, it is only a conceptual contract; actual dataset loading logic is not implemented here.

This is the place where real dataset readers would later live for different data sources or dataset types.

---

### `MODELS/`

Purpose:
- defines a generic model interface used by the pipeline.

Key file:
- `model.py`

The project defines an abstract base class `Model` with methods:

- `train()`
- `test()`
- `valid()`

This suggests the intended architecture is to separate pipeline orchestration from model implementation, leaving actual training logic to concrete model classes that would be added later.

---

### `PIPELINE/`

Purpose:
- central orchestration logic for end-to-end processing.

Key file:
- `pipeline.py`

This file is one of the central architectural pieces. It defines a `Pipeline` class and uses the config object to decide which processing stage to start from.

Important aspects:

- `PROCESS_LAYER` determines which stage is active:
  - `RAW` = 0
  - `BRONZE` = 1
  - `SILVER` = 2
  - `GOLD` = 3
- `run()` dispatches to layer-specific loaders and stage-specific processing functions:
  - `raw_loader()`
  - `bronze_loader()`
  - `silver_loader()`
  - `gold_loader()`
  - `bronze()`
  - `silver()`
  - `gold()`

Current status:
- most loader methods are placeholders (`pass`), and stage processing methods are abstract.
- this is a skeleton of a pipeline, not a completed implementation.

This is the main place to extend when building a working data-processing flow.

---

### `PROCESING/`

Purpose:
- image processing and face extraction before model input.

Key file:
- `preproc.py`

This file contains the most concrete runtime logic in the repository so far.

Core class:
- `Preproc`

Responsibilities:
- initialize a YOLO face-detection model using `ultralytics`:
  - `YOLO("TOOLS/yolov8/yolov8n-face.pt")`
- process a video stream and extract face crops
- detect the largest face in a frame
- resize it to a target square size
- compute frame-to-frame difference features and normalize them

Important methods:
- `proces(vid, recenter, square_size)`
  - reads video frames
  - detects a face every `recenter` frames
  - crops the face region
  - builds a list of face frames
  - calculates differences between adjacent frames
  - normalizes using standardization
- `face_detector(img)`
  - runs YOLO object detection
  - selects the largest face bounding box by area

This is likely the main feature extraction component for the project, especially if the goal is face-based physiological signal extraction.

Important note:
- The folder is named `PROCESING`, but the import path in `main.py` uses `PREPROC`, which indicates a naming inconsistency or an unfinished refactor.

---

### `TOOLS/`

Purpose:
- shared utilities used by the pipeline and preprocessing modules.

Files:
- `config_parser.py`
- `resampling.py`
- `__init__.py`
- `yolov8/yolov8n-face.pt` (pretrained face detector weights)

#### `config_parser.py`

This module handles YAML config loading:

- `ConfigParser.parse(path)` reads YAML and returns a Python object
- `Layer` enum defines stage values:
  - `RAW = 0`
  - `BRONZE = 1`
  - `SILVER = 2`
  - `GOLD = 3`

This is the main config contract used by the pipeline.

#### `resampling.py`

This small utility provides:

- `video_resampler(video, fs_start, fs_end)`
- `gt_resampler(signal, fs_start, fs_end)`

It uses `scipy.signal.resample_poly`, which is a practical implementation for converting data between sampling rates.

This is a general-purpose data utility that fits the signal-processing theme of the project.

---

### `TESTS/`

Purpose:
- validation of helper functionality.

Key file:
- `test_tools.py`

Covered tests:
- `test_resampling_video`
- `test_resampled_gt`

These tests validate that frame and signal resampling returns the expected length and finite values.

This is a good minimal baseline for future additions, especially after adding or refactoring pipeline logic.

---

## 4. Important files and their roles

### `main.py`

This is the likely top-level execution script.

It does the following:

1. parses CLI arguments (`--config`)
2. loads YAML configuration
3. runs the pipeline
4. references a `Preproc` object and `face_detector`

Current code:

```python
import argparse
from PIPELINE.pipeline import Pipeline
from TOOLS.config_parser import ConfigParser
from PREPROC.preproc import Preproc
```

This shows the intended runtime flow, but it also exposes naming inconsistencies and likely incomplete integration between modules.

---

### `README.md`

Currently minimal. It says:

```md
# CLNN
CLNN implemented in medallions architecture
```

It is not yet a complete developer guide. This project context document is intended to fill that gap.

---

### `pyproject.toml`

Python package metadata and dependencies.

Project metadata:
- project name: `clnn`
- Python requirement: `>=3.13`

Core dependencies include:
- `numpy`
- `opencv-python`
- `pandas`
- `torch` and `torchvision`
- `ultralytics`
- `scipy`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `requests`
- `pytest`

This indicates the project is positioned around:
- computer vision
- video processing
- data science / signal processing
- deep learning / model experimentation

---

## 5. How the project is intended to work

### Configuration-driven run

The runtime is intended to be config-driven. A YAML file supplies the run settings, such as:

```yaml
MAIN:
  PROCESS_LAYER: "raw/bronze/silver/gold"

RAW:
  PATH: "./hello/world"

BRONZE:
  PATH: "./hello/world"
  FS: 35
  DOWN_SAMPLE:
    DO: "YES"
    FS: 30
```

This strongly suggests the tool is designed to work on video data and to operate on dataset stages.

---

### Pipeline stages

The intended pipeline is sequential and layered:

1. `RAW`
   - ingest raw audio/video or dataset source
2. `BRONZE`
   - convert raw source into structured/cleaned intermediate data
3. `SILVER`
   - normalize, crop, resample, or transform for modeling
4. `GOLD`
   - final curated data ready for model training or inference

This matches the `medallions architecture` noted in the README.

---

### Preprocessing and signal extraction

The concrete feature pipeline appears to be centered around face detection and signal transformation:

- `Preproc.face_detector()` finds the largest visible face in each frame
- `Preproc.proces()` crops face regions and builds a sequence of frames
- adjacent frame differences are normalized to produce a signal-like representation
- resampling utilities then adjust the temporal sampling rate

This suggests the project is oriented toward video-based physiological or facial signal extraction, possibly remote photoplethysmography or related biometric tasks.

---

## 6. Current maturity / caveats

This project is not yet a polished end-to-end application. It currently behaves more like a research scaffold than a finished software package.

Common signs of this:

- abstract base classes with `pass` implementations
- config example file instead of real production config
- inconsistent directory naming (`PROCESING` vs expected `PREPROC`)
- `main.py` imports a module that may not exist as written
- file names suggest experimentation and unfinished refactoring

The best way to understand the project is to treat it as a prototype pipeline architecture that is still being assembled.

---

## 7. How to work with this project

### Recommended setup

Use a Python 3.13 environment and install dependencies via `uv` or `pip`.

Typical setup:

```bash
uv sync
```

or, if not using `uv`:

```bash
pip install -r <generated requirements>
```

The project metadata suggests a direct installation from the project file is the intended route.

---

### Run the application

The entry point is:

```bash
python main.py --config path/to/config.yaml
```

This expects a YAML config file that matches the structure shown in `CONFIGS/HOW_CONFIG_SHOOULD_LOOK_LIKE.yaml`.

---

### Extend the pipeline

If you want to add real functionality, the likely extension points are:

1. `CONFIGS/` — define complete real configuration files
2. `LOADER/loader.py` — implement actual dataset loading logic
3. `PIPELINE/pipeline.py` — add stage-specific data processing and orchestration
4. `PROCESING/preproc.py` — improve face detection and signal extraction
5. `MODELS/model.py` — add concrete training/inference models
6. `TESTS/` — add coverage for new functionality

---

### Validation

The repo already includes pytest-based checks for helper functionality:

```bash
pytest
```

This is the minimal validation workflow for the project in its current state.

---

## 8. Working assumptions for future contributors

The project probably expects contributors to think in terms of:

- stage-based ETL/medallion processing,
- video-based signal extraction,
- YAML-driven runtime configuration,
- modular yet incomplete orchestration,
- experimentation, research, and iterative implementation.

When working in this repo, treat it as a prototype architecture with a strong conceptual shape but incomplete implementation details.

---

## 9. Practical summary

If you were onboarding to this repository, the key mental model is:

- this is not a finished product,
- it is a pipeline skeleton for CLNN-style processing,
- data and processing go through raw -> bronze -> silver -> gold stages,
- face detection and preprocessing are the most operational parts right now,
- configuration and testing are the central conventions to build upon,
- the next meaningful step is to complete the missing abstractions and fix naming inconsistencies.

That is the best way to "work" with this project today.
