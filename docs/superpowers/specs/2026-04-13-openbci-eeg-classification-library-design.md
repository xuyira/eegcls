# OpenBCI EEG Classification Library Design

## Goal

Build a product-oriented EEG classification library for OpenBCI `.txt` files.

The library starts after data acquisition. It is responsible for:

- parsing OpenBCI `.txt` files
- extracting the first 8 EEG channels
- preprocessing and slicing data into fixed windows
- training a supervised single-label classifier
- saving a reusable model artifact
- running offline inference on a single `.txt` file
- returning per-window predictions in time order

The library is not intended to be a general research framework in v1.

## Scope

### In Scope

- Offline, non-real-time processing
- Single-label classification
- Fixed-size input windows
- Fixed dataset directory layout
- Supervised labels from directory names
- Standardized training, validation, and test flows
- Standardized model artifact format
- A single product-facing inference entrypoint

### Out of Scope

- Real-time streaming inference
- Multi-label classification
- Automatic label discovery or clustering
- Arbitrary user-defined preprocessing pipelines
- Arbitrary input formats beyond the supported OpenBCI `.txt` format
- A generic experiment management platform

## Core Product Assumptions

The library enforces strong constraints.

Only the neural network implementation is expected to vary between models. Everything else should be standardized:

- input format
- label source
- preprocessing behavior
- dataset loading
- training loop behavior
- saved artifact structure
- inference API

This is intentional. The main goal is stable training and stable reuse of trained models, not maximum experimentation flexibility.

## Dataset Contract

The dataset layout is fixed:

```text
dataset/
  train/
    <label_name>/
      *.txt
  val/
    <label_name>/
      *.txt
  test/
    <label_name>/
      *.txt
```

Rules:

- labels come from subdirectory names
- each `.txt` file belongs to exactly one split
- all windows sliced from a file inherit that file's directory label
- only the first 8 EEG channels are used
- all samples must be converted into a fixed input shape before reaching the model

### Split Safety

To avoid leakage:

- a raw `.txt` file must appear in only one of `train`, `val`, or `test`
- windows produced from one raw file must never be split across multiple dataset splits

This is mandatory because overlapping windows from the same raw signal can make validation metrics unrealistically high.

## OpenBCI File Handling

The parser is responsible for converting a raw OpenBCI `.txt` file into a clean numeric matrix.

The parser contract must define and version the following:

- whether the file contains headers or comments
- which columns correspond to EEG data
- how the first 8 EEG channels are selected
- how malformed rows are handled
- how missing values are handled
- how non-numeric values are handled
- what sampling rate is assumed or read from metadata

The parser output should be a raw tensor-like array with shape:

```text
[C, T]
```

where:

- `C = 8`
- `T = number of time points`

## Preprocessing Contract

Preprocessing is owned by the library and must be identical during training and inference.

The preprocessing pipeline should be fixed by configuration, not by ad hoc scripts.

Expected v1 preprocessing steps:

- channel selection: first 8 EEG channels
- optional bad-row cleanup
- optional resampling to the configured sampling rate
- optional notch filtering
- optional band-pass filtering
- normalization using a fixed strategy
- fixed-window slicing with a configured stride

The exact enabled steps may vary by config, but the pipeline structure and configuration format should remain stable.

### Windowing

The model input length is fixed.

Required parameters:

- `sampling_rate`
- `window_size`
- `stride`
- `short_window_policy`

`short_window_policy` should support a small fixed set of behaviors, preferably:

- `drop`
- `pad`

Every produced window must have the same shape:

```text
[8, T_window]
```

where `T_window` is constant for a trained model.

## Label Contract

Labels are supervised and derived from dataset folder names.

Example:

```text
train/
  left/
  right/
  rest/
```

This creates a frozen mapping such as:

```json
{
  "left": 0,
  "rest": 1,
  "right": 2
}
```

Requirements:

- label mapping must be generated once during training
- label mapping must be saved with the model artifact
- inference must load the saved mapping, not regenerate it from current directories

This avoids silent class-index mismatches between training and inference.

## Architecture

The library should be split into focused modules with clear boundaries.

### `io`

Responsibilities:

- read OpenBCI `.txt` files
- parse numeric EEG values
- return the raw 8-channel matrix

This layer should know nothing about training logic or model inference.

### `preprocess`

Responsibilities:

- apply configured signal preprocessing
- slice raw signals into fixed windows
- produce model-ready arrays

This layer must be deterministic from saved configuration.

### `dataset`

Responsibilities:

- scan `train/val/test` directory layouts
- associate directory names with labels
- turn files into windowed examples
- return `x`, `y`, and metadata

Suggested metadata fields:

- `source_file`
- `split`
- `label_name`
- `label_index`
- `window_index`
- `start_idx`
- `end_idx`

### `model`

Responsibilities:

- define interchangeable network backbones
- preserve a fixed input and output contract

Required model contract:

- input: `[B, 8, T]`
- output: `[B, num_classes]`

Different backbone implementations are allowed, but this contract is not.

### `trainer`

Responsibilities:

- build model from config
- run train and validation loops
- track metrics
- save the best checkpoint
- evaluate on test data when requested

The trainer should not contain OpenBCI-specific parsing logic.

### `artifact`

Responsibilities:

- package everything needed for future inference
- validate artifact completeness
- load artifacts for prediction

This module is critical because the real product value is not only training a model, but reliably reusing it later.

### `inference`

Responsibilities:

- load a saved artifact
- accept a single OpenBCI `.txt` path
- parse, preprocess, slice, and predict
- return per-window predictions in time order

## Training Behavior

Training should be standardized and intentionally narrow.

Recommended minimal training config:

- `model_name`
- `sampling_rate`
- `window_size`
- `stride`
- `channels = 8`
- `num_classes`
- `batch_size`
- `epochs`
- `optimizer`
- `learning_rate`
- `normalization`
- optional filter settings

Avoid exposing a large number of user-tunable framework knobs in v1.

The purpose of training configuration is to support stable product training, not open-ended experimentation.

## Model Artifact Format

Saving only a bare weight file is not enough.

Each trained model should be exported as a complete artifact bundle containing at least:

- `model.pt` or `model.safetensors`
- `model_config.yaml`
- `preprocess_config.yaml`
- `label_map.json`
- `train_summary.json`
- `library_version`

Optional additional files:

- `metrics.json`
- `confusion_matrix.csv`
- `class_names.json`

The artifact must contain enough information to reproduce inference behavior without rescanning the training dataset.

## Inference API

The external product-facing inference API should be intentionally simple.

Primary API:

```text
predict_file(path_to_txt) -> List[WindowPrediction]
```

The caller provides one OpenBCI `.txt` file.

The library internally performs:

1. file parsing
2. channel extraction
3. preprocessing
4. window slicing
5. model inference
6. ordered result formatting

The API should not require callers to construct tensors or understand preprocessing internals.

### Output Contract

Inference output should return one result per window in time order.

Suggested fields for each window result:

- `window_index`
- `start_idx`
- `end_idx`
- `start_time`
- `end_time`
- `pred_label`
- `pred_index`
- `confidence`
- `probabilities`

No file-level averaging or voting is needed in v1.

## Evaluation

Because the product inference output is window-level, the primary evaluation metrics should also be window-level.

Recommended metrics:

- accuracy
- macro F1
- per-class precision/recall/F1
- confusion matrix

If file-level evaluation is needed later, it can be added as a separate reporting mode, but it should not define the v1 training target.

## Error Handling

The library should fail clearly on invalid inputs.

Important failure cases:

- unsupported or malformed OpenBCI `.txt` format
- fewer than 8 usable EEG channels
- file shorter than the configured minimum window requirement
- label directories missing from one or more splits
- artifact missing required metadata files
- mismatch between artifact config and runtime expectations

Errors should describe:

- which file failed
- which step failed
- what condition was violated

Silent fallback behavior should be avoided wherever possible.

## Configuration Philosophy

v1 should prefer a small number of explicit configuration files over runtime improvisation.

Recommended config groups:

- dataset config
- preprocessing config
- model config
- training config
- inference config

All runtime behavior that affects model semantics should be representable in saved config.

## Versioning and Reproducibility

The system should make it possible to understand exactly how a model was produced.

Each artifact should store:

- library version
- model type
- preprocessing parameters
- label mapping
- core training parameters

Recommended extras:

- random seed
- training timestamp
- dataset summary counts

## Non-Goals for v1

These should remain explicitly excluded unless a later product requirement demands them:

- clustering-based pseudo-label generation such as TICC
- support for arbitrary sensor counts
- online adaptation
- model ensembling
- plugin-style user preprocessing hooks
- multi-file temporal aggregation

## Recommended v1 Build Order

1. Implement OpenBCI `.txt` parsing with strict validation.
2. Implement preprocessing and fixed-window slicing.
3. Implement dataset scanning for `train/val/test/<label>/*.txt`.
4. Implement a single baseline model backend with the fixed contract.
5. Implement training, validation, and checkpoint saving.
6. Implement artifact packaging and loading.
7. Implement `predict_file(path_to_txt)` returning ordered per-window predictions.
8. Add evaluation reports and minimal CLI or API wrappers.

## Key Risks

### Parsing Ambiguity

If the OpenBCI `.txt` format is not strictly specified, the whole system becomes fragile. This must be pinned down first.

### Data Leakage

Overlapping windows from the same raw file across different splits will invalidate metrics.

### Training/Inference Drift

If preprocessing differs between training and inference, model performance will degrade unpredictably.

### Artifact Incompleteness

If label mapping or preprocessing configuration is not stored, trained models are not safely reusable.

## Final Recommendation

Build v1 as a strongly constrained product library for offline OpenBCI EEG window classification.

The defining characteristics of the design are:

- one supported input format
- one supported task type
- one standardized preprocessing flow
- one standardized artifact format
- one simple inference entrypoint
- interchangeable model backbones behind a fixed tensor contract

This keeps the product boundary clear and makes later implementation tractable without turning the library into an unconstrained experiment framework.
