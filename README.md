# MicroDreamer

**Language-Conditioned Video Trajectory Generation and Action Prediction for Micro/Nano-Robot Manipulation**

English | [中文](./README_zh.md)

MicroDreamer is a dual-output model that simultaneously generates future video frames and predicts action sequences for micro/nano-robot manipulation tasks under a microscope. It is designed for cell manipulation operations such as ICSI (intracytoplasmic sperm injection), cell sorting, and embryo transfer.

## Architecture

### Dual-Resolution Pipeline
- **High-res path (1600×1200)**: InternViT-6B-style tile encoder → Action prediction (5-DOF diffusion)
- **Low-res path (512×384)**: CogVideoX-style temporal model → Video prediction

### Key Components
1. **TileAttentionEncoder**: Processes high-res images via 448×448 tiling (12 tiles for 4×3 grid)
2. **DiffusionActionHead**: Predicts action sequences [stage_dx, stage_dy, pip_dx, pip_dy, pip_dz]
3. **VideoPredictionModel**: Autoregressive frame prediction with LoRA fine-tuning
4. **LanguageEncoder**: Flan-T5-based text conditioning via cross-attention

## Project Structure

```
MicroDreamer/
├── config/           # Configuration management
├── hardware/         # Hardware abstraction layer
│   ├── virtual/      # Virtual devices for testing
│   ├── camera/       # Basler camera driver
│   ├── stage/        # Nikon Ti2E stage controller
│   └── pipette/      # HTTP API pipette controller
├── data/             # Data collection and preprocessing
│   ├── collector/    # Synchronized multi-device data recorder
│   ├── preprocessor/ # Frame/action preprocessing
│   ├── annotation/   # Language annotation format
│   └── dataset.py    # PyTorch Dataset
├── models/           # Model components
│   ├── action/       # Action prediction (visual encoder + diffusion head)
│   ├── video/        # Video prediction (CogVideoX + LoRA)
│   └── language/     # Language encoder
├── inference/        # Prediction pipeline
├── scripts/          # Training scripts
├── tests/            # Unit tests
└── utils/            # Utilities (logging, calibration)
```

## Quick Start

```bash
# Install dependencies
conda activate microdreamer
pip install -r requirements.txt

# Run all tests
python tests/run_all_tests.py

# Launch data collection UI
python scripts/collect_ui.py

# Or collect data via command line
python scripts/collect_data.py --mode virtual --num_episodes 10

# Train action model
python scripts/train_action.py --data_dir ./data/raw --output_dir ./outputs --simple_lang

# Train video model
python scripts/train_video.py --data_dir ./data/raw --output_dir ./outputs

# Evaluate
python scripts/evaluate.py --data_dir ./data/raw

# Inference demo
python inference/predict.py --demo
```

### Data Collection UI

The project includes a tkinter-based GUI for interactive data collection with:
- **Live camera preview** with real-time frame display
- **Stage control**: Manual XY movement with configurable step size
- **Pipette control**: XY movement + Z-axis descent/ascent
- **PID auto-positioning**: Click on camera view to set target, PID controller auto-moves stage
- **Data recording**: Start/stop recording with task descriptions

See [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md) for detailed instructions.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System architecture and module details
- [API Reference](docs/API.md) — Complete API documentation
- [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md) — 数据采集指南 (中文)
- [Data Collection Guide (EN)](docs/DATA_COLLECTION_GUIDE_EN.md) — English version
- [E2E Test Guide](docs/E2E_TEST_GUIDE.md) — 端到端测试手册 (中文)
- [E2E Test Guide (EN)](docs/E2E_TEST_GUIDE_EN.md) — English version
- [Development Log](docs/DEVELOPMENT.md) — Development progress and changelog

## Hardware Requirements
- Training: 6-8× NVIDIA 5090 GPUs
- Inference: 1-2× GPUs
- Testing: CPU-only (virtual devices)

## E2E Test Commands

```bash
# 0. Environment
conda activate microdreamer
cd D:\SimonWorkspace\MicroRobotDataGen\MicroDreamer

# 1. Generate mock data (with targets)
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160

# 1.5 Visualize data (interactive GUI)
python scripts/viz_mock_data.py

# 2. Run all unit tests (~34 tests)
python tests/run_all_tests.py

# 3. Train action model (3 epochs, test config)
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3

# 4. Train video model (3 epochs, test config)
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3

# 5. Evaluate both models
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml

# 5.5 Visualize evaluation results
python scripts/viz_mock_data.py --data_dir ./data/test_raw

# 6. Inference
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"

# 6.5 Visualize inference results
python scripts/viz_mock_data.py --data_dir ./data/test_raw

# 7. Clean up
rmdir /s /q data\test_raw
rmdir /s /q outputs\test
```

Quick demo (no checkpoint needed):
```bash
python inference/predict.py --demo --device cpu
```

Static visualization (generates PNGs):
```bash
python scripts/visualize_mock_data.py --output_dir ./data/viz_mock --save_dir ./outputs/viz
```

See [E2E Test Guide](docs/E2E_TEST_GUIDE.md) for full details.

## License
TBD
