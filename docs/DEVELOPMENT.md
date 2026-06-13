# MicroDreamer Development Log

## Phase 1: Project Framework (2026-06-13)

### Summary
Built complete project framework with all core modules, virtual hardware testing, and full test suite.

### Modules Implemented

#### 1. Hardware Abstraction Layer (`hardware/`)
- **Base classes**: `CameraBase`, `StageBase`, `PipetteBase` with abstract interfaces
- **Virtual devices**: `VirtualCamera` (synthetic frames with moving circle), `VirtualStage`, `VirtualPipette`
- **Real drivers**: `BaslerCamera` (pypylon), `NikonStage` (DLL ctypes), `HttpPipette` (REST API)
- **Factory**: `create_camera/stage/pipette()` from config

#### 2. Data Pipeline (`data/`)
- **Synchronizer**: Multi-threaded camera/stage/pipette collection with timestamp alignment
- **Recorder**: Save episodes as `.npz` + `metadata.json`
- **Preprocessor**: `positions_to_deltas()` action conversion, `ActionNormalizer`, `tile_frame()`, `resize_frame()`
- **Dataset**: `MicroDreamerDataset` PyTorch Dataset with dual-res output
- **Annotation**: Task templates (ICSI, cell sorting, embryo transfer)

#### 3. Models (`models/`)
- **Visual Encoder**: `TileAttentionEncoder` (InternViT-style, 12 tiles × 448×448), `LowResEncoder`
- **Action Head**: `DiffusionActionHead` (8-layer cross-attention), `ActionDiffusion` (DDPM)
- **Action Model**: `ActionPredictionModel` (visual + language → diffusion actions)
- **Video Model**: `VideoPredictionModel` (CogVideoX-style, temporal transformer, LoRA)
- **Language**: `SimpleLanguageEncoder` (testing), `LanguageEncoder` (Flan-T5)
- **Losses**: `VideoLoss` (L1 + temporal consistency)

#### 4. Inference (`inference/`)
- **MicroDreamerPredictor**: Unified predictor for both action and video prediction

#### 5. Training Scripts (`scripts/`)
- `train_action.py`: Action model training with diffusion loss
- `train_video.py`: Video model training with reconstruction loss

#### 6. Configuration (`config/`)
- YAML-based config with dot-access
- Hardware, preprocessing, model, training sections

#### 7. Utilities (`utils/`)
- Logger with file + console output
- `PixelCalibration` (pixel ↔ um), `FocusCalibration` (sharpness-based Z estimation)

### Test Results
```
5 test suites, 22 individual tests - ALL PASSED
- VirtualCamera, VirtualStage, VirtualPipette
- Action conversion, normalization, frame resize/tiling
- Dataset creation and loading
- TileAttentionEncoder, LowResEncoder, DiffusionActionHead, ActionDiffusion
- LanguageEncoder, full ActionPredictionModel
- VideoPredictionModel forward/no-language, VideoLoss
```

### Bugs Fixed
1. **PatchEmbedding non-square**: Added tuple support for `tile_size` parameter
2. **Video decoder resolution**: Added `AdaptiveAvgPool2d` to match target resolution
3. **Inference tile flattening**: Removed manual flattening (model handles internally)
4. **Inference frame dims**: Added extra `unsqueeze` for 5D video input

### Git History
- `main`: Stable, all tests passing
- `dev`: Active development branch
- Commits: initial plan → network test → full framework → cleanup
