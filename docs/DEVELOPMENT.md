# MicroDreamer Development Log

## Phase 1: Project Framework (2026-06-13)

### Summary
Built complete project framework with all core modules, virtual hardware testing, and full test suite.

### Modules Implemented
- **Hardware abstraction**: Virtual + real device drivers (Basler camera, Nikon stage, HTTP pipette)
- **Data pipeline**: Synchronized collection, preprocessing, PyTorch Dataset, annotation format
- **Models**: TileAttentionEncoder, DiffusionActionHead, VideoPredictionModel, LanguageEncoder
- **Inference**: Unified MicroDreamerPredictor
- **Config**: YAML-based configuration system
- **Utils**: Logging, calibration (pixel-to-um, focus depth)

### Test Results
5 test suites, 22 tests - ALL PASSED

---

## Phase 2: Training Pipeline & Tools (2026-06-13)

### Summary
Complete training infrastructure with TensorBoard, evaluation metrics, calibration, and bug fixes.

### New Features

#### Training (`scripts/`)
- **train_action.py**: Action model training with:
  - TensorBoard logging (loss curves, learning rate)
  - Gradient clipping (max_norm=1.0)
  - Cosine LR scheduler with linear warmup
  - Checkpoint save/resume with full optimizer/scheduler state
  - Early stopping (configurable patience)
  - Mixed precision (FP16)
  - Per-dimension action metrics during training

- **train_video.py**: Video model training with:
  - Separate LR for LoRA params (5x higher)
  - Sample image visualization in TensorBoard
  - L1 + temporal consistency loss
  - Same robustness features as action training

#### Evaluation (`scripts/evaluate.py`)
- Action metrics: MSE, MAE, per-dimension MSE, endpoint error, trajectory length, consistency
- Video metrics: pixel MSE/MAE, PSNR, SSIM, temporal consistency, FVD
- JSON output for experiment tracking

#### Calibration (`scripts/calibrate.py`)
- Pixel-to-micrometer calibration
- Focus-depth calibration via sharpness analysis
- JSON calibration file output

#### Data Collection (`scripts/collect_data.py`)
- Virtual mode: scripted movements (linear/circular/zigzag)
- Real mode: interactive collection with manual control
- Episode-based recording with metadata

#### Metrics (`models/action/metrics.py`, `models/video/metrics.py`)
- ActionMetrics: accumulates batches, computes aggregate stats
- VideoMetrics: pixel-level and perceptual metrics

### Bug Fixes
1. **SimpleLanguageEncoder**: Now accepts `text=` keyword (auto-encodes to token ids)
2. **Video model resolution**: Fixed (H, W) convention mismatch with config [W, H]
3. **Action model config**: Reduced hidden_dim/layers for GPU memory (512/6 vs 1024/12)
4. **LR scheduler**: Moved step to epoch level to avoid per-batch stepping
5. **LoRA implementation**: Proper LoRALinear/LoRAMultiheadAttention with merge support

### LoRA Implementation
- LoRALinear: freezes original weights, adds low-rank A×B delta
- LoRAMultiheadAttention: separate Q/K/V with LoRA
- get_lora_params() / get_non_lora_params() for optimizer groups
- merge_lora() for inference acceleration
- ~10% of total parameters are LoRA-trainable

### Training Test Results
- Action model: 100 epochs, loss decreasing, ~1.3s/epoch (GPU)
- Video model: 100 epochs, loss decreasing, ~0.4s/epoch (GPU)
- All 22 unit tests passing

### Git History
- `dev` branch: active development
- `main` branch: stable, tested
- 8 commits in Phase 2

---

## Phase 3: Data Collection UI & Documentation (2026-06-13)

### Summary
Added interactive data collection GUI, PID auto-positioning, comprehensive data collection documentation, and bug fixes.

### New Features

#### Data Collection UI (`scripts/collect_ui.py`)
- **tkinter-based GUI** for interactive microscope data collection
- **Live camera preview** with real-time frame display (30fps)
- **Stage control panel**: Manual XY movement with configurable step size
- **Pipette control panel**: XY movement + Z-axis descent/ascent
- **PID auto-positioning**: Click on camera view to set target, PID controller auto-moves stage
  - Configurable Kp, Ki, Kd parameters
  - Anti-windup integral clamping
  - Output limiting for safety
- **Data recording**: Start/stop recording with task descriptions
- **Episode management**: Browse and review collected episodes
- **Status bar**: Real-time FPS, position display, recording status

#### Data Collection Documentation
- **DATA_COLLECTION_GUIDE.md** (中文): 详细的数据采集指南
  - 硬件环境配置
  - 采集前准备流程
  - 数据采集步骤详解
  - 标注规范（语言标注模板）
  - 数据格式说明
  - 质量控制检查清单
  - 常见问题与故障排除
- **DATA_COLLECTION_GUIDE_EN.md** (English): Full English version

#### Bug Fixes
1. **Inference dimension mismatch**: SimpleLanguageEncoder hidden_dim now matches video model context_dim
2. **Inference argparse**: Added proper --help support and demo mode
3. **PID controller**: Proper anti-windup with integral clamping

### Test Results
- 7 test suites, 31 tests - ALL PASSED
- New test suite: `test_scripts/test_collect_ui.py` (PID controller + DataCollector)

### Updated Files
- `inference/predict.py`: Fixed lang_encoder dimension, added argparse
- `scripts/collect_ui.py`: New file - data collection GUI
- `tests/test_scripts/test_collect_ui.py`: New file - UI component tests
- `tests/run_all_tests.py`: Added new test suites
- `README.md`: Added UI documentation and data collection guide links
- `README_zh.md`: Added Chinese UI documentation and guide links
- `docs/DATA_COLLECTION_GUIDE.md`: New file - Chinese data collection guide
- `docs/DATA_COLLECTION_GUIDE_EN.md`: New file - English data collection guide

---

## Phase 4: End-to-End Pipeline Testing & Bug Fixes (2026-06-14)

### New Features

#### Test Data Generation
- **`scripts/generate_test_data.py`**: Generate mock training data for testing
  - Configurable resolution, episode count, frame count
  - Realistic moving circle patterns with multiple objects
  - Smooth stage/pipette trajectories
  - Proper metadata.json with task descriptions and subgoals
  - 5 task templates (ICSI, cell sorting, embryo transfer, zona drilling, aspirate)

#### Test Configuration
- **`config/test.yaml`**: Small/fast test config for CI/quick verification
  - 200x160 resolution, 128 hidden_dim, 2 layers, 4 heads
  - 3 epochs, batch_size=2, fp16=false, deepspeed=false
  - Compatible with CPU-only testing

#### End-to-End Pipeline Test
- **`tests/test_e2e_pipeline.py`**: Comprehensive E2E test suite (7 tests)
  1. Mock data generation — structure verification
  2. Dataset loading — shape and format checks
  3. Action model training — 2 epochs, loss decrease verification
  4. Video model training — 2 epochs with LoRA, checkpoint save/load
  5. Action evaluation — metrics computation
  6. Video evaluation — PSNR, SSIM, temporal consistency
  7. Inference — action + video prediction shape verification

#### Documentation
- **`docs/E2E_TEST_GUIDE.md`**: 端到端测试手册 (中文)
  - 8 个步骤的完整测试流程
  - CMD 和 Bash 两个终端的命令
  - 预期结果和验证方法
  - 故障排除指南
  - 测试配置参数对比
- **`docs/E2E_TEST_GUIDE_EN.md`**: English version

### Bug Fixes

1. **evaluate.py resolution bug**: Video model resolution was passed as `(W, H)` instead of `(H, W)`. Fixed to match train_video.py's approach.

2. **evaluate.py missing num_tiles**: Action model was created without `num_tiles` from config, causing shape mismatch with non-default configs. Fixed.

3. **evaluate.py tile reshape bug**: Tiles were incorrectly reshaped to `(B*T, 1, 1, H, W)` instead of passing `(B, T, 1, H, W)` directly to the model. Fixed.

4. **evaluate.py missing lora params**: Video model was created without `lora_rank`/`lora_alpha` from config. Fixed.

5. **predict.py checkpoint format**: Checkpoint was loaded directly as state_dict but training wraps it in `{"model": state_dict, ...}`. Fixed to handle both formats.

6. **predict.py missing num_tiles**: Action model created without `num_tiles` from config. Fixed.

7. **predict.py resolution bug**: Video model resolution passed as `(W, H)` instead of `(H, W)`. Fixed.

8. **predict.py missing lora params**: Video model created without `lora_rank`/`lora_alpha`. Fixed.

9. **PyTorch 2.6+ torch.load compatibility**: All `torch.load()` calls now include `weights_only=False` for PyTorch 2.6+ compatibility. Fixed in: evaluate.py, train_action.py, train_video.py, predict.py, test_e2e_pipeline.py.

### Test Results
- 8 test suites, ~38 tests - ALL PASSED
- New test suite: `test_e2e_pipeline.py` (7 tests)
- Full pipeline verified: generate → train → evaluate → inference

### Updated Files
- `scripts/generate_test_data.py`: New file - mock data generation
- `config/test.yaml`: New file - test configuration
- `tests/test_e2e_pipeline.py`: New file - E2E pipeline tests
- `tests/run_all_tests.py`: Added E2E pipeline test
- `scripts/evaluate.py`: Fixed resolution, num_tiles, tile reshape, lora params
- `inference/predict.py`: Fixed checkpoint loading, resolution, num_tiles, lora params
- `scripts/train_action.py`: Fixed torch.load compatibility
- `scripts/train_video.py`: Fixed torch.load compatibility
- `docs/E2E_TEST_GUIDE.md`: New file - Chinese test guide
- `docs/E2E_TEST_GUIDE_EN.md`: New file - English test guide
- `README.md`: Added E2E test guide links
- `README_zh.md`: Added E2E test guide links
