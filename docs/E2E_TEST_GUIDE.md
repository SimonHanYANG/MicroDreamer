# MicroDreamer 端到端测试手册

## 概述

本手册提供完整的端到端测试流程，验证 MicroDreamer 系统的所有核心功能：

1. **数据生成** — 生成模拟训练数据
2. **数据加载** — 验证 Dataset 加载和预处理
3. **动作模型训练** — 训练动作预测模型
4. **视频模型训练** — 训练视频预测模型
5. **模型评估** — 评估两个模型的指标
6. **推理预测** — 使用训练好的模型进行推理

---

## 环境准备

### 1. 激活 conda 环境

```cmd
conda activate microdreamer
```

### 2. 进入项目目录

```cmd
cd D:\SimonWorkspace\MicroRobotDataGen\MicroDreamer
```

### 3. 验证环境

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 测试流程

### 步骤 1：生成模拟测试数据

生成 10 个 episode 的模拟数据，用于后续所有测试。

**CMD:**
```cmd
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**Bash:**
```bash
python scripts/generate_test_data.py --output_dir ./data/test_raw --num_episodes 10 --frames 50 --resolution 200,160
```

**验证结果：**
- 应看到 `Done! Generated 10 episodes in data\test_raw`
- 验证数据格式：
  ```cmd
  python -c "import numpy as np; d=np.load('./data/test_raw/episode_20260614_000000_0000/data.npz'); print('frames:', d['frames'].shape, 'stage:', d['stage_positions'].shape, 'pipette:', d['pipette_positions'].shape)"
  ```
- 预期输出：`frames: (50, 160, 200) stage: (50, 2) pipette: (50, 3)`

---

### 步骤 2：运行单元测试（快速验证）

运行所有单元测试套件，确认基础功能正常。

**CMD:**
```cmd
python tests/run_all_tests.py
```

**Bash:**
```bash
python tests/run_all_tests.py
```

**验证结果：**
- 应看到 `Results: 8 passed, 0 failed out of 8 tests`
- 8 个测试套件包括：
  - `test_virtual_devices` — 虚拟硬件设备
  - `test_preprocessor` — 数据预处理
  - `test_dataset` — 数据集加载
  - `test_action_model` — 动作模型组件
  - `test_video_model` — 视频模型组件
  - `test_e2e` — 集成测试
  - `test_collect_ui` — 数据采集 UI 组件
  - `test_e2e_pipeline` — 完整流水线测试

---

### 步骤 3：训练动作预测模型

使用测试配置训练动作预测模型（3 个 epoch）。

**CMD:**
```cmd
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_action.py --data_dir ./data/test_raw --output_dir ./outputs/test --simple_lang --config config/test.yaml --patience 3
```

**验证结果：**
- 应看到 `Device: cuda`（或 `cpu`）
- 应看到 `Model params: 1.3M`
- 每个 epoch 显示 loss 值
- 最后应显示 `Training complete`
- 检查 checkpoint 文件：
  ```cmd
  dir outputs\test\checkpoints\
  ```
  应有 `action_best.pt` 和 `action_ckpt_epoch*.pt`

---

### 步骤 4：训练视频预测模型

使用测试配置训练视频预测模型（3 个 epoch）。

**CMD:**
```cmd
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**Bash:**
```bash
python scripts/train_video.py --data_dir ./data/test_raw --output_dir ./outputs/test --config config/test.yaml --patience 3
```

**验证结果：**
- 应看到 `Video model params: 5.4M`
- 每个 epoch 显示 loss 值
- 最后应显示 `Training complete`
- 检查 checkpoint：
  ```cmd
  dir outputs\test\checkpoints\
  ```
  应有 `video_best.pt`

---

### 步骤 5：评估模型

使用训练好的 checkpoint 评估两个模型。

**CMD:**
```cmd
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**Bash:**
```bash
python scripts/evaluate.py --data_dir ./data/test_raw --output_dir ./outputs/test/eval --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml
```

**验证结果：**

动作模型评估指标：
- `action_mse` — 动作均方误差（应为有限数值）
- `action_mae` — 动作平均绝对误差
- `endpoint_error` — 终点误差
- `mse_stage_dx`, `mse_stage_dy` — Stage XY 方向 MSE
- `mse_pip_dx`, `mse_pip_dy`, `mse_pip_dz` — Pipette XYZ 方向 MSE

视频模型评估指标：
- `pixel_mse` — 像素均方误差
- `pixel_mae` — 像素平均绝对误差
- `psnr` — 峰值信噪比
- `ssim` — 结构相似性
- `temporal_consistency` — 时间一致性

检查输出文件：
```cmd
type outputs\test\eval\eval_action.json
type outputs\test\eval\eval_video.json
```

---

### 步骤 6：运行推理

使用训练好的模型进行推理预测。

**CMD:**
```cmd
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**Bash:**
```bash
python inference/predict.py --action_ckpt ./outputs/test/checkpoints/action_best.pt --video_ckpt ./outputs/test/checkpoints/video_best.pt --config config/test.yaml --device cpu --task "Aspirate the target cell"
```

**验证结果：**
- 应显示 `Predictor loaded on cpu`

**Demo 模式（无需 checkpoint）：**

```cmd
python inference/predict.py --demo --device cpu
```

**验证结果：**
- 应显示 `Actions shape: (16, 5)`
- 应显示 `Predicted frames shape: (4, 1, 384, 512)`

---

### 步骤 7：数据采集 UI（可选）

启动数据采集 GUI 界面。

**CMD:**
```cmd
python scripts/collect_ui.py
```

**Bash:**
```bash
python scripts/collect_ui.py
```

**验证结果：**
- 应弹出 tkinter 窗口
- 包含：摄像头预览、Stage 控制、Pipette 控制、PID 控制、录制面板

---

### 步骤 8：清理测试数据

测试完成后清理生成的临时文件。

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

## 预期耗时

| 步骤 | 预期耗时 | 说明 |
|------|---------|------|
| 1. 生成数据 | ~5 秒 | 10 episodes, 200x160 分辨率 |
| 2. 单元测试 | ~60 秒 | 8 个测试套件, ~38 个测试 |
| 3. 训练动作模型 | ~15 秒 | 3 epochs, GPU |
| 4. 训练视频模型 | ~10 秒 | 3 epochs, GPU |
| 5. 评估模型 | ~90 秒 | 动作+视频评估 |
| 6. 推理 | ~30 秒 | 含模型加载 |
| **总计** | **~3-4 分钟** | 全部步骤 |

> 注：CPU 模式下训练和推理耗时约为 GPU 的 3-5 倍。

---

## 故障排除

### 问题 1：`ModuleNotFoundError: No module named 'xxx'`

**解决方案：** 确保已激活 conda 环境并安装所有依赖：
```cmd
conda activate microdreamer
pip install -r requirements.txt
```

### 问题 2：`torch.load` 报错 `WeightsUnpickler error`

**原因：** PyTorch 2.6+ 默认 `weights_only=True`

**解决方案：** 已在代码中修复，所有 `torch.load` 调用已添加 `weights_only=False`

### 问题 3：CUDA 内存不足 (OOM)

**解决方案：** 使用 `config/test.yaml` 已配置小模型（128 hidden_dim, 2 layers），应可在任何 GPU 上运行。如仍 OOM，设置 `--device cpu`。

### 问题 4：评估时 `resolution` 不匹配

**原因：** 配置文件中 resolution 为 `[W, H]` 格式，模型内部使用 `(H, W)` 格式

**解决方案：** 已在代码中修复，evaluate.py 和 predict.py 已正确转换

### 问题 5：`num_tiles` 不匹配

**原因：** 测试配置使用 `num_tiles: 1`（小分辨率），但某些脚本未从配置读取

**解决方案：** 已在代码中修复，所有模型创建点已正确传入 `num_tiles` 配置

---

## 测试配置说明

使用 `config/test.yaml` 而非 `config/default.yaml`，参数对比：

| 参数 | default.yaml | test.yaml | 说明 |
|------|-------------|-----------|------|
| camera resolution | 1600x1200 | 200x160 | 小分辨率加速测试 |
| hidden_dim | 512 | 128 | 小模型加速 |
| num_layers | 6 | 2 | 更少层 |
| num_heads | 8 | 4 | 更少头 |
| action_horizon | 16 | 4 | 更短预测 |
| num_frames | 16 | 8 | 更少帧 |
| max_epochs | 100 | 3 | 快速验证 |
| batch_size | 2 | 2 | 保持不变 |
| fp16 | true | false | CPU 兼容 |
| deepspeed | true | false | 单 GPU 测试 |

---

## 所有脚本 `--help` 命令速查

```cmd
python scripts/generate_test_data.py --help
python scripts/collect_data.py --help
python scripts/collect_ui.py --help
python scripts/train_action.py --help
python scripts/train_video.py --help
python scripts/evaluate.py --help
python scripts/calibrate.py --help
python inference/predict.py --help
```
