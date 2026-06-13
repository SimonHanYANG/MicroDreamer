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
