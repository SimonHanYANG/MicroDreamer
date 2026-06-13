# MicroDreamer API Reference

## Hardware Layer

### `hardware.base.CameraBase`
Abstract camera interface.
- `open() -> None`
- `capture() -> Frame` (image: np.ndarray, timestamp: float, frame_id: int)
- `close() -> None`
- `is_open() -> bool`
- `resolution -> (width, height)`
- `fps -> float`

### `hardware.base.StageBase`
Abstract XY stage interface.
- `connect() -> None`
- `get_position() -> Position2D` (x, y in um)
- `move_absolute(x, y) -> None`
- `move_relative(dx, dy) -> None`
- `disconnect() -> None`

### `hardware.base.PipetteBase`
Abstract 3-DOF pipette interface.
- `connect() -> None`
- `get_position() -> Position3D` (x, y, z in um)
- `move_absolute(x, y, z) -> None`
- `move_relative(dx, dy, dz) -> None`
- `disconnect() -> None`

### `hardware.factory`
- `create_camera(cfg) -> CameraBase`
- `create_stage(cfg) -> StageBase`
- `create_pipette(cfg) -> PipetteBase`

## Data Layer

### `data.collector.synchronizer.DataSynchronizer`
Multi-threaded data collection with timestamp alignment.
```python
sync = DataSynchronizer(camera, stage, pipette, camera_fps=30)
sync.start()
sample = sync.get_latest_sample()  # SyncedSample
sync.stop()
```

### `data.collector.recorder.DataRecorder`
Episode-based data recording.
```python
recorder = DataRecorder("./data/raw")
episode_id = recorder.start_episode("ICSI injection")
recorder.save_samples(samples)
```

### `data.dataset.MicroDreamerDataset`
PyTorch Dataset for training.
```python
ds = MicroDreamerDataset("./data/raw", action_horizon=16, low_res=(512, 384))
sample = ds[0]
# Returns: high_res_tiles, low_res_frames, actions, task_description
```

### `data.preprocessor.action_converter`
- `positions_to_deltas(stage_pos, pip_pos) -> actions (T-1, 5)`
- `deltas_to_positions(init_stage, init_pip, actions) -> (stage_pos, pip_pos)`
- `ActionNormalizer`: fit/normalize/denormalize/save/load

### `data.preprocessor.frame_processor`
- `tile_frame(frame, tile_size=448) -> tiles (N, 448, 448)`
- `resize_frame(frame, (W, H)) -> resized`
- `normalize_frame(frame) -> float32 [0, 1]`

### `data.annotation.format`
- `EpisodeAnnotation`: episode_id, task_description, subgoals, num_frames
- `TASK_TEMPLATES`: predefined task types (icsi, cell_sorting, embryo_transfer)
- `create_annotation(episode_id, task_type, num_frames) -> EpisodeAnnotation`

## Model Layer

### `models.action.visual_encoder.TileAttentionEncoder`
InternViT-style visual encoder with tile-based processing.
```python
encoder = TileAttentionEncoder(tile_size=448, embed_dim=512, num_layers=6, num_heads=8)
tiles = torch.randn(B, 12, 1, 448, 448)
features = encoder(tiles)  # (B, embed_dim)
```

### `models.action.action_head.DiffusionActionHead`
Diffusion-based action sequence predictor.
```python
head = DiffusionActionHead(action_dim=5, action_horizon=16, hidden_dim=512)
noise_pred = head(noisy_actions, timestep, context)  # (B, T, 5)
```

### `models.action.action_model.ActionPredictionModel`
Complete action prediction model.
```python
model = ActionPredictionModel(hidden_dim=512, use_simple_lang=True)
loss = model.training_loss(tiles, gt_actions, lang_text=["inject sperm"])
actions = model.predict_actions(tiles, lang_text=["inject sperm"])  # (B, 16, 5)
```

### `models.video.video_model.VideoPredictionModel`
Video prediction with LoRA fine-tuning.
```python
model = VideoPredictionModel(hidden_dim=256, resolution=(384, 512), lora_rank=64)
pred_frames = model(input_frames, lang_context=ctx, num_pred=4)  # (B, 4, C, H, W)
model.merge_lora()  # for faster inference
```

### `models.video.video_model.LoRALinear`
Low-Rank Adaptation for linear layers.
```python
lora_layer = LoRALinear.from_linear(original_linear, rank=64, alpha=128)
output = lora_layer(x)  # original(x) + lora_delta(x)
```

### `models.language.encoder.SimpleLanguageEncoder`
Lightweight language encoder for testing.
```python
encoder = SimpleLanguageEncoder(vocab_size=1000, hidden_dim=512)
embeddings = encoder(text=["move to cell"])  # (B, L, 512)
```

## Metrics

### `models.action.metrics.ActionMetrics`
```python
metrics = ActionMetrics()
metrics.update(pred_actions, gt_actions)
results = metrics.compute()
# Returns: action_mse, action_mae, endpoint_error, mse_per_dim, ...
```

### `models.video.metrics.VideoMetrics`
```python
metrics = VideoMetrics()
metrics.update(pred_frames, gt_frames)
results = metrics.compute()
# Returns: pixel_mse, pixel_mae, psnr, ssim, temporal_consistency
```

## Inference

### `inference.predict.MicroDreamerPredictor`
Unified predictor for both tasks.
```python
predictor = MicroDreamerPredictor(action_ckpt="best.pt", device="cpu")
result = predictor.predict(frame, task_description="aspirate cell")
# Returns: {"actions": (16, 5), "predicted_frames": (4, C, H, W)}
```

## Configuration

### `config.config.load_config`
```python
cfg = load_config("config/default.yaml")
cfg.camera.resolution       # [1600, 1200]
cfg.action_model.hidden_dim # 512
cfg.training.learning_rate  # 1e-4
```

## Utilities

### `utils.calibration.PixelCalibration`
```python
cal = PixelCalibration(pixel_size_um=0.6)
um_x, um_y = cal.pixel_to_um(100, 50)  # 60.0, 30.0
```

### `utils.calibration.FocusCalibration`
```python
cal = FocusCalibration()
cal.calibrate(images, z_positions)
estimated_z = cal.estimate_z(image)
```

### `utils.logger.setup_logger`
```python
logger = setup_logger("my_module", level=logging.INFO, log_dir="logs")
```
