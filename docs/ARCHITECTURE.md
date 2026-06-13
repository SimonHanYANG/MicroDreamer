# MicroDreamer Architecture

## System Overview

MicroDreamer is a dual-output model for micro/nano-robot manipulation under a microscope. Given a high-resolution image and a language instruction, it simultaneously predicts:
1. **Action sequence** (5-DOF): stage dx/dy, pipette dx/dy/dz
2. **Future video frames**: predicted visual outcomes

## Dual-Resolution Pipeline

```
Input Frame (1600×1200)
        │
        ├──→ High-res Path (1600×1200, 3072 tokens)
        │    └──→ TileAttentionEncoder (12 tiles × 448×448)
        │         └──→ DiffusionActionHead → Actions (T, 5)
        │
        └──→ Low-res Path (512×384, 256 tokens)
             └──→ VideoPredictionModel (temporal transformer + LoRA)
                  └──→ Predicted Frames (T, C, H, W)

Language Instruction → Flan-T5 / SimpleLanguageEncoder → Cross-Attention
```

## Module Architecture

### 1. Visual Encoder (`models/action/visual_encoder.py`)

**TileAttentionEncoder** (InternViT-style):
- Splits 1600×1200 image into 12 tiles of 448×448 (4×3 grid)
- Each tile → PatchEmbedding (14×14 patches, 1024 patches/tile) → Transformer blocks
- Tile position embeddings encode spatial layout
- Cross-attention aggregation: learnable query attends to all tile CLS tokens
- Output: single feature vector (B, hidden_dim)

**LowResEncoder**:
- 512×384 input → PatchEmbedding (16×16 patches, 768 patches)
- Transformer blocks → CLS token as global feature

### 2. Language Encoder (`models/language/encoder.py`)

**LanguageEncoder** (production):
- Flan-T5 encoder pretrained on instruction-following data
- Linear projection from T5 dim → model hidden_dim
- Optional freezing for efficient fine-tuning

**SimpleLanguageEncoder** (testing):
- Character-level vocabulary embedding
- Learnable positional encoding
- LayerNorm output
- Accepts raw text strings via `text=` parameter

### 3. Action Head (`models/action/action_head.py`)

**DiffusionActionHead**:
- Input: noisy actions (B, T, 5) + timestep embedding
- 8 cross-attention layers: self-attn → cross-attn (with visual+language context) → FFN
- Output: predicted noise for DDPM denoising

**ActionDiffusion**:
- Standard DDPM with linear beta schedule (100 timesteps)
- Forward: q_sample adds noise to ground truth actions
- Reverse: p_sample iteratively denoises
- Full sampling: 100-step reverse process

### 4. Video Model (`models/video/video_model.py`)

**VideoPredictionModel** (CogVideoX-style):
- Frame encoder: Conv → global average pool → feature per frame
- Temporal transformer: LoRA-wrapped attention over frame sequence
- Autoregressive: predict one frame, append, repeat
- Frame decoder: Linear → deconv → AdaptiveAvgPool to target resolution

**LoRA** (Low-Rank Adaptation):
- LoRALinear: freezes original weights, adds low-rank delta (A×B)
- LoRAMultiheadAttention: separate LoRA on Q, K, V projections
- ~10% of parameters are trainable LoRA params
- merge_lora() for inference acceleration

### 5. Data Pipeline (`data/`)

```
Camera (30Hz) ──┐
Stage (100Hz) ──┼──→ DataSynchronizer ──→ SyncedSample ──→ DataRecorder ──→ .npz
Pipette (100Hz)─┘                          (timestamp-aligned)

data.npz:
  frames: (T, H, W) uint8
  stage_positions: (T, 2) float32  [x, y] in um
  pipette_positions: (T, 3) float32 [x, y, z] in um
  timestamps: (T,) float64

metadata.json:
  episode_id, task_description, subgoals[], num_frames
```

**Preprocessing**:
- `positions_to_deltas()`: absolute → relative action deltas
- `ActionNormalizer`: zero-mean unit-variance normalization
- `tile_frame()`: 1600×1200 → 12 tiles of 448×448
- `resize_frame()`: bilinear interpolation via scipy.ndimage.zoom

### 6. Hardware Abstraction (`hardware/`)

```
CameraBase ← VirtualCamera | BaslerCamera (pypylon)
StageBase  ← VirtualStage  | NikonStage (DLL ctypes)
PipetteBase← VirtualPipette| HttpPipette (REST API)

factory.py: create_camera/stage/pipette(cfg) → device instance
```

## Training Pipeline

### Action Model Training
```
Dataset → DataLoader → batch{tiles, actions, text}
    → encode_context(tiles, text) → visual_feat + lang_feat
    → diffusion.q_sample(gt_actions, t, noise) → noisy_actions
    → action_head(noisy_actions, t, context) → pred_noise
    → MSE(pred_noise, noise) → loss
    → gradient clipping → optimizer → scheduler
```

### Video Model Training
```
Dataset → DataLoader → batch{frames, text}
    → input_frames[:4], target_frames[4:8]
    → model(input_frames, lang_context, num_pred=4) → pred_frames
    → L1(pred, target) + temporal_consistency(pred) → loss
    → LoRA params get 5x higher learning rate
```

## Configuration System

YAML-based hierarchical config with dot-access:
```python
cfg = load_config("config/default.yaml")
cfg.camera.resolution      # [1600, 1200]
cfg.action_model.hidden_dim # 512
cfg.training.learning_rate  # 1e-4
```

Supports command-line overrides and environment-specific configs.
