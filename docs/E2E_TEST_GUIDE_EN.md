# MicroDreamer End-to-End Test Guide

## Overview

This guide provides a complete end-to-end testing procedure to verify all core functionalities of the MicroDreamer system:

1. **Data Generation** — Generate mock training data
2. **Data Visualization** — Interactively inspect generated data quality
3. **Data Loading** — Verify Dataset loading and preprocessing
4. **Action Model Training** — Train the action prediction model
5. **Video Model Training** — Train the video prediction model
6. **Model Evaluation** — Evaluate both models with metrics
7. **Inference** — Run predictions with trained models

> The visualization tool `viz_mock_data.py` can be used at any stage — after data generation, before training, after evaluation, or after inference — to inspect data quality and model outputs.

---

## Environment Setup

### 1. Activate conda environment

```cmd
conda activate microdreamer
```

### 2. Navigate to project directory

```cmd
cd D:\SimonWorkspace\MicroRobotDataGen\MicroDreamer
```

### 3. Verify environment

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Test Procedure

### Step 1: Generate Mock Test Data

Generate 10 episodes of mock data for all subsequent tests.

**CMD:**
```cmd
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**Bash:**
```bash
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**Expected result:**
- Should see `Done! Generated 10 episodes in data\test_raw`
- Verify data format:
  ```cmd
  python -c "import numpy as np; d=np.load('./data/test_raw/episode_20260614_000000_0000/data.npz'); print('frames:', d['frames'].shape, 'stage:', d['stage_positions'].shape, 'pipette:', d['pipette_positions'].shape)"
  ```
- Expected output: `frames: (50, 160, 200) stage: (50, 2) pipette: (50, 3)`

---

### Step 1.5: Visualize Generated Data (Recommended)

Use the interactive visualization tool to inspect the generated mock data.

**Interactive GUI (Recommended):**
```cmd
python scripts/viz_mock_data.py
```

**Features:**
- Episode browser with dropdown selector
- Frame-by-frame playback of microscopy images
- Stage XY trajectory plot (colour-coded by subgoal)
- Stage/Pipette positions vs time
- 5-DOF action deltas
- Subgoal timeline, frame statistics
- All plots synchronised with frame slider

**What to check:**
- Frames contain identifiable moving objects (white circles)
- Stage trajectory is a continuous smooth path
- Pipette Z axis shows a descending trend (simulating aspiration)
- Subgoal time segments are reasonably divided

**Static visualization (generates PNGs):**
```cmd
python scripts/visualize_mock_data.py --output_dir ./data/viz_mock --save_dir ./outputs/viz
```
Check `episode_XX_overview.png` and `episode_XX_montage.png` in `./outputs/viz/`.

---

### Step 2: Run Unit Tests (Quick Verification)

Run all unit test suites to confirm basic functionality.

**CMD:**
```cmd
python tests/run_all_tests.py
```

**Bash:**
```bash
python tests/run_all_tests.py
```

**Expected result:**
- Should see `Results: 8 passed, 0 failed out of 8 tests`
- 8 test suites include:
  - `test_virtual_devices` — Virtual hardware devices
  - `test_preprocessor` — Data preprocessing
  - `test_dataset` — Dataset loading
  - `test_action_model` — Action model components
  - `test_video_model` — Video model components
  - `test_e2e` — Integration tests
  - `test_collect_ui` — Data collection UI components
  - `test_e2e_pipeline` — Full pipeline test

---

### Step 3: Train Action Prediction Model

Train the action prediction model with test config (3 epochs).

**CMD:**
```cmd
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**Expected result:**
- Should see `Device: cuda` (or `cpu`)
- Should see `Model params: 1.3M`
- Each epoch shows loss value
- Should end with `Training complete`
- Check checkpoint files:
  ```cmd
  dir outputs\test\checkpoints\
  ```
  Should contain `action_best.pt` and `action_ckpt_epoch*.pt`

---

> **💡 Visualization tip:** Before training, you can reopen `python scripts/viz_mock_data.py` to confirm the action delta distribution of training data is reasonable.

---

### Step 4: Train Video Prediction Model

Train the video prediction model with test config (3 epochs).

**CMD:**
```cmd
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**Expected result:**
- Should see `Video model params: 5.4M`
- Each epoch shows loss value
- Should end with `Training complete`
- Check checkpoint:
  ```cmd
  dir outputs\test\checkpoints\
  ```
  Should contain `video_best.pt`

---

### Step 5: Evaluate Models

Evaluate both models using trained checkpoints.

**CMD:**
```cmd
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**Bash:**
```bash
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**Expected result:**

Action model metrics:
- `action_mse` — Action mean squared error (should be finite)
- `action_mae` — Action mean absolute error
- `endpoint_error` — Endpoint displacement error
- `mse_stage_dx`, `mse_stage_dy` — Stage XY direction MSE
- `mse_pip_dx`, `mse_pip_dy`, `mse_pip_dz` — Pipette XYZ direction MSE

Video model metrics:
- `pixel_mse` — Pixel mean squared error
- `pixel_mae` — Pixel mean absolute error
- `psnr` — Peak signal-to-noise ratio
- `ssim` — Structural similarity
- `temporal_consistency` — Temporal consistency

Check output files:
```cmd
type outputs\test\eval\eval_action.json
type outputs\test\eval\eval_video.json
```

---

### Step 5.5: Visualize Evaluation Results (Recommended)

After evaluation, visually compare predictions against ground truth.

**Interactive inspection:**
```cmd
python scripts/viz_mock_data.py --data_dir ./data/test_raw
```
Browse through the original data frame by frame, and cross-reference with metrics in `outputs/test/eval/eval_action.json` and `eval_video.json`.

**What to check:**
- `action_mse` / `endpoint_error` are finite (not NaN/Inf)
- Video metric `psnr` > 20dB (may be lower with test config, which is normal)
- `temporal_consistency` close to 1.0 indicates good temporal coherence

---

### Step 6: Run Inference

Run inference with trained models.

**CMD:**
```cmd
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**Bash:**
```bash
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**Expected result:**
- Should display `Predictor loaded on cpu`

**Demo mode (no checkpoint needed):**

```cmd
python inference/predict.py --demo --device cpu
```

**Expected result:**
- Should display `Actions shape: (16, 5)`
- Should display `Predicted frames shape: (4, 1, 384, 512)`

---

### Step 6.5: Visualize Inference Results (Recommended)

After inference, use the visualization tool to compare input frames with predicted outputs.

**Interactive inspection of inference inputs:**
```cmd
python scripts/viz_mock_data.py --data_dir ./data/test_raw
```
Select any episode and play through to observe model input data:
- Are the frame contents clear?
- Are the position trajectories smooth?
- Are the action delta magnitudes reasonable?

**What to check:**
- Inference output `Actions shape: (16, 5)` — 5-DOF action sequence
- Predicted frames `shape: (4, 1, 384, 512)` — 4 future video frames
- Compare ground truth trajectory in `viz_mock_data.py` with inference output for the same episode

---

### Step 7: Data Collection UI (Optional)

Launch the data collection GUI.

**CMD:**
```cmd
python scripts/collect_ui.py
```

**Bash:**
```bash
python scripts/collect_ui.py
```

**Expected result:**
- A tkinter window should appear
- Contains: Camera preview, Stage control, Pipette control, PID control, Recording panel

---

### Step 8: Clean Up Test Data

Clean up generated temporary files after testing.

**CMD:**
```cmd
rmdir /s /q data\test_raw
rmdir /s /q outputs\test
```

**Bash:**
```bash
rm -rf ./data/test_raw ./outputs/test
```

---

## Expected Duration

| Step | Expected Time | Notes |
|------|--------------|-------|
| 1. Generate data | ~5 sec | 10 episodes, 200x160 resolution |
| 1.5. Visualize data | Manual | `viz_mock_data.py` interactive, optional |
| 2. Unit tests | ~60 sec | 8 test suites, ~38 tests |
| 3. Train action model | ~15 sec | 3 epochs, GPU |
| 4. Train video model | ~10 sec | 3 epochs, GPU |
| 5. Evaluate models | ~90 sec | Action + video evaluation |
| 5.5. Visualize eval | Manual | Cross-check metrics with data, optional |
| 6. Inference | ~30 sec | Including model loading |
| 6.5. Visualize inference | Manual | Compare predictions vs ground truth, optional |
| **Total** | **~3-4 min** | Automated steps only (excluding visualization) |

> Note: CPU mode takes approximately 3-5x longer for training and inference.

---

## Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'xxx'`

**Solution:** Ensure conda environment is activated and all dependencies installed:
```cmd
conda activate microdreamer
pip install -r requirements.txt
```

### Issue 2: `torch.load` error `WeightsUnpickler error`

**Cause:** PyTorch 2.6+ defaults to `weights_only=True`

**Solution:** Fixed in code — all `torch.load` calls now include `weights_only=False`

### Issue 3: CUDA out of memory (OOM)

**Solution:** The `config/test.yaml` uses small model (128 hidden_dim, 2 layers), should run on any GPU. If still OOM, use `--device cpu`.

### Issue 4: `resolution` mismatch during evaluation

**Cause:** Config files use `[W, H]` format, model internally uses `(H, W)` format

**Solution:** Fixed in code — evaluate.py and predict.py now correctly convert resolution

### Issue 5: `num_tiles` mismatch

**Cause:** Test config uses `num_tiles: 1` (small resolution), but some scripts didn't read from config

**Solution:** Fixed in code — all model creation points now correctly pass `num_tiles` from config

---

## Test Config Comparison

Using `config/test.yaml` instead of `config/default.yaml`:

| Parameter | default.yaml | test.yaml | Description |
|-----------|-------------|-----------|-------------|
| camera resolution | 1600x1200 | 200x160 | Small resolution for speed |
| hidden_dim | 512 | 128 | Smaller model |
| num_layers | 6 | 2 | Fewer layers |
| num_heads | 8 | 4 | Fewer heads |
| action_horizon | 16 | 4 | Shorter prediction |
| num_frames | 16 | 8 | Fewer frames |
| max_epochs | 100 | 3 | Quick verification |
| batch_size | 2 | 2 | Same |
| fp16 | true | false | CPU compatible |
| deepspeed | true | false | Single GPU test |

---

## All Scripts `--help` Quick Reference

```cmd
python scripts/generate_test_data.py --help
python scripts/collect_data.py --help
python scripts/collect_ui.py --help
python scripts/train_action.py --help
python scripts/train_video.py --help
python scripts/evaluate.py --help
python scripts/calibrate.py --help
python inference/predict.py --help
python scripts/viz_mock_data.py --help
python scripts/visualize_mock_data.py --help
```
